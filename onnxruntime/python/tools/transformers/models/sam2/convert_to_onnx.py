# -------------------------------------------------------------------------
# Copyright (R) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Example commands to export onnx:
#   git clone https://github.com/facebookresearch/segment-anything-2.git
#   cd segment-anything-2
#   pip install -e .
#   cd checkpoints
#   sh ./download_ckpts.sh
#   cd ..
#   wget https://raw.githubusercontent.com/microsoft/onnxruntime/main/onnxruntime/python/tools/transformers/models/sam2/convert_to_onnx.py
#   python convert_to_onnx.py

from typing import Any, Optional

import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base
from torch import nn


class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, x: torch.Tensor) -> tuple[Any, Any, Any]:
        """
        Encodes images into features.

        Args:
            x (torch.Tensor): images of shape [B, 3, H, W], B is batch size, H and W are height and width.

        Returns:
            image_features_0: image features of shape [B, 32, H/4, W/4] - high resolution features of level 0
            image_features_1: image features of shape [B, 64, H/8, W/8] - high resolution features of level 1
            image_embeddings: image features of shape [B, 256, H/16, W/16] - 16 is the backbone_stride
        """
        backbone_out = self.image_encoder(x)

        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])

        # Prepare and flatten visual features.
        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.num_feature_levels :]
        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # flatten NxCxHxW to HWxNxC
        # TODO: we should avoid this transpose since it will be transposed back to NCHW later.
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]

        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        return feats[0], feats[1], feats[2]


class SAM2ImageDecoder(nn.Module):
    def __init__(
        self,
        sam_model: SAM2Base,
        multimask_output: bool,
        dynamic_multimask_via_stability: bool = True,
        output_low_res_masks: bool = True,
    ) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.multimask_output = multimask_output
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.output_low_res_masks = output_low_res_masks

    @torch.no_grad()
    def forward(
        self,
        image_features_0: torch.Tensor,
        image_features_1: torch.Tensor,
        image_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        input_masks: Optional[torch.Tensor] = None,
    ):
        """Decode masks from image features and prompts.

           Limitations:
            (1) Only support image size HxW=1024x1024. If you want to use different image sizes like 512x512,
                see https://github.com/facebookresearch/segment-anything-2/issues/138.
            (2) Batched images are not supported.

           Args:
            image_features_0 (torch.Tensor): [B, 32, H/4, W/4]. high resolution features of level 0 from image encoder.
            image_features_1 (torch.Tensor): [B, 64, H/8, W/8]. high resolution features of level 1 from image encoder.
            image_embeddings (torch.Tensor): [B, 256, H/16, W/16]. image embedding from image encoder.
            point_coords (torch.Tensor): [B, P, 2] shape and float32 dtype and contains the absolute pixel
                                         coordinate in (x, y) format of the P input points.
            point_labels (torch.Tensor): shape [B, P] and int32 dtype, where 1 means
                                         positive (foreground), 0 means negative (background), and -1 means padding.
            input_masks (torch.Tensor, optional): [B, 1, H/4, W/4]. Low resolution mask input to the model.
                                        Typically coming from a previous iteration.
        Returns:
            masks (torch.Tensor): [B, M, H, W] where M=3 or 1. masks of image_size.
            iou_predictions (torch.Tensor): [B, M]. scores for M masks.
            low_res_masks (torch.Tensor, optional): [B, M, H/4, W/4]. low resolution masks.
        """
        # Boxes shall be converted and concanetate with points, see sam2_image_onnx_predictor.py for details.
        boxes = None
        sparse_embeddings = self.prompt_encoder._embed_points(point_coords, point_labels, pad=(boxes is None))
        low_res_masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=self.embed_masks(input_masks),
            repeat_image=False,
            high_res_features=[image_features_0, image_features_1],
        )

        if self.multimask_output:
            low_res_masks = low_res_masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
        elif self.dynamic_multimask_via_stability:
            # When outputting a single mask, if the stability score from the current single-mask
            # output (based on output token 0) falls below a threshold, we instead select from
            # multi-mask outputs (based on output token 1~3) the mask with the highest predicted IoU score.
            low_res_masks, iou_predictions = self.mask_decoder._dynamic_multimask_via_stability(
                low_res_masks, iou_predictions
            )
        else:
            low_res_masks = low_res_masks[:, 0:1, :, :]
            iou_predictions = iou_predictions[:, 0:1]

        # Compute the original image height and width, then interpolate the low resolution masks to the image size.
        masks = F.interpolate(
            low_res_masks,
            (image_features_0.shape[2] * 4, image_features_0.shape[3] * 4),
            mode="bilinear",
            align_corners=False,
        )

        if self.output_low_res_masks:
            low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
            return masks, iou_predictions, low_res_masks
        else:
            return masks, iou_predictions

    def embed_masks(self, masks: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Embeds mask inputs. This function is not used in onnx export. It's used to demonstrate
        how to prepare mask embeddings as input for onnx model.

        Args:
            masks: shape [B, 1, 256, 256]. Low resolution mask input to the model,
                   each has shape 1xHxW, where H=W=256. Typically coming from a previous iteration.
        """
        if masks is None:
            return self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        else:
            return self.prompt_encoder.mask_downscaling(masks)


def get_model_cfg(model_type):
    assert model_type in ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"]
    if model_type == "sam2_hiera_tiny":
        model_cfg = "sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small":
        model_cfg = "sam2_hiera_s.yaml"
    elif model_type == "sam2_hiera_base_plus":
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = "sam2_hiera_l.yaml"
    return model_cfg


def export_encoder_onnx(
    model_type,
    dynamic_batch_axes=True,  # Set to False if only support batch size 1.
    verbose=False,
):
    # image_size is configured in yaml file (like sam2_hiera_l.yaml)
    image_height = 1024
    image_width = 1024

    sam2_checkpoint = f"checkpoints/{model_type}.pt"

    model_cfg = get_model_cfg(model_type)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")

    image = torch.randn(1, 3, image_height, image_width).cpu()

    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    image_features_0, image_features_1, image_embeddings = sam2_encoder(image)
    if verbose:
        print("image.shape", image.shape)
        print("image_features_0.shape", image_features_0.shape)
        print("image_features_1.shape", image_features_1.shape)
        print("image_embeddings.shape", image_embeddings.shape)

    dynamic_axes = None
    if dynamic_batch_axes:
        dynamic_axes = {
            "image": {0: "batch_size"},
            "image_features_0": {0: "batch_size"},
            "image_features_1": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
        }

    onnx_model_path = f"{model_type}_encoder.onnx"
    torch.onnx.export(
        sam2_encoder,
        image,
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["image_features_0", "image_features_1", "image_embeddings"],
        dynamic_axes=dynamic_axes,
    )

    print("encoder onnx model saved to ", onnx_model_path)


def export_decoder_onnx(
    model_type,
    multimask_output=False,
    has_mask_input=False,
    output_low_res_masks=False,
    dynamic_batch_axes=True,  # Set to False will force batch size to be 1.
    verbose=True,
):
    # image_size is configured in yaml file (like sam2_hiera_l.yaml)
    image_height = 1024
    image_width = 1024

    sam2_checkpoint = f"checkpoints/{model_type}.pt"
    model_cfg = get_model_cfg(model_type)
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cpu")
    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()

    # Run encoder to get image features and embeddings.
    batch_size = 1
    image = torch.randn(batch_size, 3, image_height, image_width).cpu()
    image_features_0, image_features_1, image_embeddings = sam2_encoder(image)

    # Enable output_low_res_masks, and run an example inputs to get low resolution masks.
    sam2_decoder = SAM2ImageDecoder(
        sam2_model, multimask_output=multimask_output, dynamic_multimask_via_stability=True, output_low_res_masks=True
    ).cpu()

    point_coords = torch.randint(low=0, high=min(image_height, image_width), size=(batch_size, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(batch_size, 5), dtype=torch.float)
    input_masks = None

    masks, scores, low_res_masks = sam2_decoder(
        image_features_0,
        image_features_1,
        image_embeddings,
        point_coords,
        point_labels,
        input_masks,
    )

    if verbose:
        print("masks.shape", masks.shape)
        print("scores.shape", scores.shape)
        print("low_res_masks.shape", low_res_masks.shape)

    if has_mask_input:
        # When mulitmask_output is True, we only use the first low_res_masks for input_masks.
        input_masks = low_res_masks[:, 0:1, :, :]

    example_inputs = (
        image_features_0,
        image_features_1,
        image_embeddings,
        point_coords,
        point_labels,
        input_masks,
    )

    input_names = [
        "image_features_0",
        "image_features_1",
        "image_embeddings",
        "point_coords",
        "point_labels",
        "input_masks",
    ]

    if not has_mask_input:
        example_inputs = example_inputs[:-1]
        input_names = input_names[:-1]

    output_names = ["masks", "scores"]
    if output_low_res_masks:
        output_names.append("low_res_masks")

    sam2_decoder.output_low_res_masks = output_low_res_masks

    onnx_model_path = (
        f"{model_type}_decoder"
        + ("_mask_in" if has_mask_input else "")
        + ("_multi_out" if multimask_output else "")
        + ".onnx"
    )

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }
    if dynamic_batch_axes:
        dynamic_axes = {
            "image_features_0": {0: "batch_size"},
            "image_features_1": {0: "batch_size"},
            "image_embeddings": {0: "batch_size"},
            "point_coords": {0: "batch_size", 1: "num_points"},
            "point_labels": {0: "batch_size", 1: "num_points"},
            "input_masks": {0: "batch_size"},
            "masks": {0: "batch_size"},
            "iou_predictions": {0: "batch_size"},
            "low_res_masks": {0: "batch_size"},
        }

    torch.onnx.export(
        sam2_decoder,
        example_inputs,
        onnx_model_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    print("decoder onnx model saved to ", onnx_model_path)


def main():
    # Here we export all combinations. User can choose models based on application.
    # for model_type in ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"]:
    for model_type in ["sam2_hiera_large"]:
        # export_encoder_onnx(model_type)
        for multimask_output in [False, True]:
            for has_mask_input in [False, True]:
                export_decoder_onnx(model_type, multimask_output=multimask_output, has_mask_input=has_mask_input)


if __name__ == "__main__":
    main()
