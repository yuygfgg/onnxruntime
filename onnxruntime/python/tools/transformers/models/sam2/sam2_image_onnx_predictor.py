# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Segment Anything v2 image predictor using ONNX models

import logging
from typing import Optional, Tuple, Union
import os
import numpy as np
import torch
from PIL.Image import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.sam2_base import SAM2Base
from sam2.utils.transforms import SAM2Transforms

from onnxruntime import InferenceSession
from onnxruntime.transformers.io_binding_helper import CudaSession
from convert_to_onnx import SAM2ImageDecoder

def create_ort_session(
    onnx_path: str,
    session_options=None,
    provider="CUDAExecutionProvider",
    enable_cuda_graph=False,
    use_tf32=True,
    prefer_nhwc=True,
) -> InferenceSession:
    if provider == "CUDAExecutionProvider":
        device_id = torch.cuda.current_device()
        provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
        provider_options["use_tf32"] = int(use_tf32)
        provider_options["prefer_nhwc"] = int(prefer_nhwc)
        providers = [(provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    print(f"Using providers: {providers}")
    return InferenceSession(onnx_path, session_options, providers=providers)


def encoder_shape_dict(batch_size: int, height: int, width: int):
    assert height == 1024 and width == 1024, "Only 1024x1024 images are supported."
    return {
        "image": [batch_size, 3, height, width],

        "image_features_0": [batch_size, 32, height // 4, width // 4],
        "image_features_1": [batch_size, 64, height // 8, width // 8],
        "image_embeddings": [batch_size, 256, height // 16, width // 16],
    }


def decoder_shape_dict(batch_size: int, height: int=1024, width: int=1024, max_points:int=16, num_masks:int=1):
    assert height == 1024 and width == 1024, "Only 1024x1024 images are supported."
    return {
        "image_features_0": [batch_size, 32, height // 4, width // 4],
        "image_features_1": [batch_size, 64, height // 8, width // 8],
        "image_embeddings": [batch_size, 256, height // 16, width // 16],
        "point_coords": [batch_size, max_points, 2],
        "point_labels": [batch_size, max_points],
        "input_masks": [batch_size, 1, height // 4, width // 4],
        "has_input_masks": [batch_size],

        "masks": [batch_size, num_masks, height, width],
        "iou_predictions": [batch_size, num_masks],
        "low_res_masks": [batch_size, num_masks, height // 4, width // 4],
    }

def create_session(
    onnx_path: str, session_options=None, provider="CUDAExecutionProvider", device="cuda", enable_cuda_graph=False
) -> CudaSession:
    ort_session = create_ort_session(
        onnx_path, session_options, provider, enable_cuda_graph=enable_cuda_graph, use_tf32=True, prefer_nhwc=True
    )
    cuda_session = CudaSession(ort_session, device=torch.device(device), enable_cuda_graph=enable_cuda_graph)
    return cuda_session


class SAM2ImageOnnxPredictor(SAM2ImagePredictor):
    def __init__(
        self,
        sam_model: SAM2Base,
        onnx_directory: str = ".",
        model_type:str = "sam2_hiera_large",
        onnx_dtype: torch.dtype = torch.float32,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        **kwargs,
    ) -> None:
        """
        Uses SAM-2 to compute the image embedding for an image, and then allow mask prediction given prompts.

        Arguments:
          sam_model (SAM2Base): The model to use for mask prediction.
          onnx_directory (str): The path of the directory that contains encoder and decoder onnx models.
          onnx_dtype (torch.dtype): The data type to use for ONNX inputs.
          mask_threshold (float): The threshold to convert mask logits to binary masks. Default is 0.0.
          max_hole_area (float): If max_hole_area > 0, we fill small holes in up to
                                 the maximum area of max_hole_area in low_res_masks.
          max_sprinkle_area (float): If max_sprinkle_area > 0, we remove small sprinkles up to
                                     the maximum area of max_sprinkle_area in low_res_masks.
        """
        super().__init__(sam_model,
                         mask_threshold=mask_threshold,
                         max_hole_area=max_hole_area,
                         max_sprinkle_area=max_sprinkle_area)

        print(self.device)

        # These onnx models shall be exported by convert_to_onnx.py in advance.
        onnx_path = os.path.join(onnx_directory, f"{model_type}_encoder.onnx")
        self.encoder_session = create_session(
            onnx_path,
            session_options=None,
            provider="CUDAExecutionProvider",
            device="cuda",
            enable_cuda_graph=False,
        )
        self.onnx_dtype = onnx_dtype

        # This is the torch class used in onnx export. We use it to compare with onnx runtime output.
        self.sam2_decoder = SAM2ImageDecoder(
            sam_model, multimask_output=False, dynamic_multimask_via_stability=True, output_low_res_masks=True,
            support_mask_input=True
        ).to(self.device)

        # Support input mask, only one mask output
        # provider = "CUDAExecutionProvider"
        # device = "cuda"
        provider = "CPUExecutionProvider"
        device = "cpu"
        onnx_path = os.path.join(onnx_directory, f"{model_type}_decoder.onnx")
        self.decoder_session = create_session(
            onnx_path,
            session_options=None,
            provider=provider,
            device=device,
            enable_cuda_graph=False,
        )

        # No input mask, only one mask output
        # onnx_path = os.path.join(onnx_directory, f"{model_type}_decoder_no-mask-in.onnx")
        # self.decoder_session_no_input_mask = create_session(
        #     onnx_path,
        #     session_options=None,
        #     provider="CUDAExecutionProvider",
        #     device="cuda",
        #     enable_cuda_graph=False,
        # )

        # Support input mask, 3 masks output
        onnx_path = os.path.join(onnx_directory, f"{model_type}_decoder_multi-mask-out.onnx")
        self.decoder_session_multi_out = create_session(
            onnx_path,
            session_options=None,
            provider=provider,
            device=device,
            enable_cuda_graph=False,
        )

        # No input mask, 3 masks output
        # onnx_path = os.path.join(onnx_directory, f"{model_type}_decoder_no-mask-in_multi-mask-out.onnx")
        # self.decoder_session_no_input_mask_multi_out = create_session(
        #     onnx_path,
        #     session_options=None,
        #     provider="CUDAExecutionProvider",
        #     device="cuda",
        #     enable_cuda_graph=False,
        # )

    @torch.no_grad()
    def set_image(self, image: Union[np.ndarray, Image]):
        """
        Calculates the image embeddings for the provided image.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format.
              The image should be in HWC format if np.ndarray, or WHC format if PIL Image with pixel values in [0, 255].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            # For numpy array image, we assume (HxWxC) format.
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        input_image = self._transforms(image)
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"

        # Computing image embeddings for the provided image
        io_shapes = encoder_shape_dict(batch_size=1, height=input_image.shape[2], width=input_image.shape[3])
        self.encoder_session.allocate_buffers(io_shapes)

        feed_dict = { "image": input_image.to(self.onnx_dtype).to(self.device) }

        for key, value in feed_dict.items():
            print(f"{key}: {value.shape}, {value.dtype}")
        print(self.encoder_session.ort_session._model_path)

        ort_outputs = self.encoder_session.infer(feed_dict)

        self._features = {
            "image_embed": ort_outputs["image_embeddings"],
            "high_res_feats": [ort_outputs[f"image_features_{i}"] for i in range(2)],
        }
        self._is_image_set = True
        logging.info("Image embeddings computed.")

    # def predict(
    #     self,
    #     point_coords: Optional[np.ndarray] = None,
    #     point_labels: Optional[np.ndarray] = None,
    #     box: Optional[np.ndarray] = None,
    #     mask_input: Optional[np.ndarray] = None,
    #     multimask_output: bool = True,
    #     return_logits: bool = False,
    #     normalize_coords=True,
    # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Predict masks for the given input prompts, using the currently set image.

    #     Arguments:
    #       point_coords (np.ndarray or None): A Nx2 array of point prompts to the
    #         model. Each point is in (X,Y) in pixels.
    #       point_labels (np.ndarray or None): A length N array of labels for the
    #         point prompts. 1 indicates a foreground point and 0 indicates a
    #         background point.
    #       box (np.ndarray or None): A length 4 array given a box prompt to the
    #         model, in XYXY format.
    #       mask_input (np.ndarray): A low resolution mask input to the model, typically
    #         coming from a previous prediction iteration. Has form 1xHxW, where
    #         for SAM, H=W=256.
    #       multimask_output (bool): If true, the model will return three masks.
    #         For ambiguous input prompts (such as a single click), this will often
    #         produce better masks than a single prediction. If only a single
    #         mask is needed, the model's predicted quality score can be used
    #         to select the best mask. For non-ambiguous prompts, such as multiple
    #         input prompts, multimask_output=False can give better results.
    #       return_logits (bool): If true, returns un-thresholded masks logits
    #         instead of a binary mask.
    #       normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1] then scaled
    #                                to image size used in model training.

    #     Returns:
    #       (np.ndarray): The output masks in CxHxW format, where C is the
    #         number of masks, and (H, W) is the original image size.
    #       (np.ndarray): An array of length C containing the model's
    #         predictions for the quality of each mask.
    #       (np.ndarray): An array of shape CxHxW, where C is the number
    #         of masks and H=W=256. These low resolution logits can be passed to
    #         a subsequent iteration as mask input.
    #     """
    #     if not self._is_image_set:
    #         raise RuntimeError(
    #             "An image must be set with .set_image(...) before mask prediction."
    #         )

    #     # Transform input prompts
    #     mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
    #         point_coords, point_labels, box, mask_input, normalize_coords
    #     )

    #     masks, iou_predictions, low_res_masks = self._predict(
    #         unnorm_coords,
    #         labels,
    #         unnorm_box,
    #         mask_input,
    #         multimask_output,
    #         return_logits=return_logits,
    #     )

    #     masks_np = masks.squeeze(0).float().detach().cpu().numpy()
    #     iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
    #     low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
    #     return masks_np, iou_predictions_np, low_res_masks_np

    # def _prep_prompts(
    #     self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    # ):

    #     unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
    #     if point_coords is not None:
    #         assert (
    #             point_labels is not None
    #         ), "point_labels must be supplied if point_coords is supplied."
    #         point_coords = torch.as_tensor(
    #             point_coords, dtype=torch.float, device=self.device
    #         )
    #         unnorm_coords = self._transforms.transform_coords(
    #             point_coords, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
    #         )
    #         labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
    #         if len(unnorm_coords.shape) == 2:
    #             unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
    #     if box is not None:
    #         box = torch.as_tensor(box, dtype=torch.float, device=self.device)
    #         unnorm_box = self._transforms.transform_boxes(
    #             box, normalize=normalize_coords, orig_hw=self._orig_hw[img_idx]
    #         )  # Bx2x2
    #     if mask_logits is not None:
    #         mask_input = torch.as_tensor(
    #             mask_logits, dtype=torch.float, device=self.device
    #         )
    #         if len(mask_input.shape) == 3:
    #             mask_input = mask_input[None, :, :, :]
    #     return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        batch_size = concat_points[0].shape[0]
        shape_dict = decoder_shape_dict(batch_size=batch_size, num_masks=3 if multimask_output else 1)
        # if mask_input is None:
        #     del shape_dict["input_masks"]
        #     del shape_dict["has_input_masks"]

        # if multimask_output:
        #     decoder_session = self.decoder_session_no_input_mask_multi_out if mask_input is None else self.decoder_session_multi_out
        # else:
        #     decoder_session = self.decoder_session_no_input_mask if mask_input is None else self.decoder_session
        if multimask_output:
            decoder_session = self.decoder_session_multi_out
        else:
            decoder_session = self.decoder_session

        decoder_session.allocate_buffers(shape_dict)

        image_features_0 = self._features["high_res_feats"][0][img_idx].unsqueeze(0)
        image_features_1 = self._features["high_res_feats"][1][img_idx].unsqueeze(0)
        image_embeddings = self._features["image_embed"][img_idx].unsqueeze(0)

        if mask_input is None:
            input_masks = torch.zeros(1, 1, 256, 256, dtype=torch.float, device=self.device)
            has_input_masks = torch.zeros(1, dtype=torch.float, device=self.device)
        else:
            input_masks = mask_input[img_idx].unsqueeze(0)
            has_input_masks = torch.ones(1, dtype=torch.float, device=self.device)

        # This is the torch one used in onnx export.
        self.sam2_decoder.multimask_output = multimask_output
        torch_low_res_masks, torch_iou_predictions  = self.sam2_decoder(
            image_features_0.clone(),
            image_features_1.clone(),
            image_embeddings.clone(),
            concat_points[0].clone(),
            concat_points[1].clone(),
            input_masks.clone(),
            has_input_masks.clone()
        )

        # "high_res_feats_0": [batch_size, 32, height // 4, width // 4],
        # "high_res_feats_1": [batch_size, 64, height // 8, width // 8],
        # "image_embeddings": [batch_size, 256, height // 16, width // 16],
        # "point_coords": [batch_size, max_points, 2],
        # "point_labels": [batch_size, max_points],
        # "input_masks": [batch_size, 1, height // 4, width // 4],
        device = "cpu"
        feed_dict = {"image_embeddings": image_embeddings.to(device),
                     "image_features_0": image_features_0.to(device),
                     "image_features_1": image_features_1.to(device),
                     "point_coords": concat_points[0].to(device),
                     "point_labels": concat_points[1].to(device),
                     "input_masks": input_masks.to(device),
                     "has_input_masks": has_input_masks.to(device),
        }
        # if mask_input is None:
        #     del feed_dict["input_masks"]
        #     del feed_dict["has_input_masks"]

        # "masks": [batch_size, num_masks, height, width],
        # "iou_predictions": [batch_size, num_masks, height // 4, width // 4],
        # "low_res_masks": [batch_size, num_masks, height // 4, width // 4],
        for key, value in feed_dict.items():
            print(f"{key}: {value.shape}, {value.dtype}")
        print(self.encoder_session.ort_session._model_path)

        ort_outputs = decoder_session.infer(feed_dict)
        print(ort_outputs)
        #masks = ort_outputs["masks"]
        iou_predictions = ort_outputs["iou_predictions"]
        low_res_masks = ort_outputs["low_res_masks"]
        # torch.testing.assert_close(masks, torch_masks.to(device), atol=1e-3, rtol=1e-3)
        #torch.testing.assert_close(iou_predictions, torch_iou_predictions.to(device), atol=1e-3, rtol=1e-3)
        #torch.testing.assert_close(low_res_masks, torch_low_res_masks.to(device), atol=1e-3, rtol=1e-3)

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(low_res_masks, self._orig_hw[img_idx])
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)

        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    # def get_image_embedding(self) -> torch.Tensor:
    #     """
    #     Returns the image embeddings for the currently set image, with
    #     shape 1xCxHxW, where C is the embedding dimension and (H,W) are
    #     the embedding spatial dimension of SAM (typically C=256, H=W=64).
    #     """
    #     if not self._is_image_set:
    #         raise RuntimeError(
    #             "An image must be set with .set_image(...) to generate an embedding."
    #         )
    #     assert (
    #         self._features is not None
    #     ), "Features must exist if an image has been set."
    #     return self._features["image_embed"]

    # @property
    # def device(self) -> torch.device:
    #     return self.model.device

    # def reset_predictor(self) -> None:
    #     """
    #     Resets the image embeddings and other state variables.
    #     """
    #     self._is_image_set = False
    #     self._features = None
    #     self._orig_hw = None
    #     self._is_batch = False
