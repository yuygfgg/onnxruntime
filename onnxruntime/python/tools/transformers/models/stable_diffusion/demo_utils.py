# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Modified from TensorRT demo diffusion, which has the following license:
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import argparse
from io import BytesIO
from typing import Any, Dict

import requests
import torch
from diffusion_models import PipelineInfo
from engine_builder import EngineType, get_engine_paths
from PIL import Image


class RawTextArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def arg_parser(description: str):
    return argparse.ArgumentParser(description=description, formatter_class=RawTextArgumentDefaultsHelpFormatter)


def parse_arguments(is_xl: bool, parser):
    engines = ["ORT_CUDA", "ORT_TRT", "TRT"]

    parser.add_argument(
        "--engine",
        type=str,
        default=engines[0],
        choices=engines,
        help="Backend engine in {engines}. "
        "ORT_CUDA is CUDA execution provider; ORT_TRT is Tensorrt execution provider; TRT is TensorRT",
    )

    supported_versions = PipelineInfo.supported_versions(is_xl)
    parser.add_argument(
        "--version",
        type=str,
        default=supported_versions[-1] if is_xl else "1.5",
        choices=supported_versions,
        help="Version of Stable Diffusion" + (" XL." if is_xl else "."),
    )

    parser.add_argument(
        "--height",
        type=int,
        default=1024 if is_xl else 512,
        help="Height of image to generate (must be multiple of 8).",
    )
    parser.add_argument(
        "--width", type=int, default=1024 if is_xl else 512, help="Height of image to generate (must be multiple of 8)."
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDIM",
        choices=["DDIM", "UniPC", "LCM"] if is_xl else ["DDIM", "EulerA", "UniPC", "LCM"],
        help="Scheduler for diffusion process" + " of base" if is_xl else "",
    )

    parser.add_argument(
        "--work-dir",
        default=".",
        help="Root Directory to store torch or ONNX models, built engines and output images etc.",
    )

    parser.add_argument("prompt", nargs="*", default=[""], help="Text prompt(s) to guide image generation.")

    parser.add_argument(
        "--negative-prompt", nargs="*", default=[""], help="Optional negative prompt(s) to guide the image generation."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8, 16],
        help="Number of times to repeat the prompt (batch size multiplier).",
    )

    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=30 if is_xl else 50,
        help="Number of denoising steps" + (" in base." if is_xl else "."),
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=5.0 if is_xl else 7.5,
        help="Higher guidance scale encourages to generate images that are closely linked to the text prompt.",
    )

    parser.add_argument(
        "--lora-scale", type=float, default=1, help="Scale of LoRA weights, default 1 (must between 0 and 1)"
    )
    parser.add_argument("--lora-weights", type=str, default="", help="LoRA weights to apply in the base model")

    if is_xl:
        parser.add_argument(
            "--lcm",
            action="store_true",
            help="Use fine-tuned latent consistency model to replace the UNet in base.",
        )

        parser.add_argument(
            "--refiner-scheduler",
            type=str,
            default="DDIM",
            choices=["DDIM", "UniPC"],
            help="Scheduler for diffusion process of refiner.",
        )

        parser.add_argument(
            "--refiner-guidance",
            type=float,
            default=5.0,
            help="Guidance scale used in refiner.",
        )

        parser.add_argument(
            "--refiner-steps",
            type=int,
            default=30,
            help="Number of denoising steps in refiner. Note that actual refiner steps is refiner_steps * strength.",
        )

        parser.add_argument(
            "--strength",
            type=float,
            default=0.3,
            help="A value between 0 and 1. The higher the value less the final image similar to the seed image.",
        )

    # ONNX export
    parser.add_argument(
        "--onnx-opset",
        type=int,
        default=None,
        choices=range(14, 18),
        help="Select ONNX opset version to target for exported models.",
    )
    parser.add_argument(
        "--force-onnx-export", action="store_true", help="Force ONNX export of CLIP, UNET, and VAE models."
    )
    parser.add_argument(
        "--force-onnx-optimize", action="store_true", help="Force ONNX optimizations for CLIP, UNET, and VAE models."
    )

    # Framework model ckpt
    parser.add_argument(
        "--framework-model-dir",
        default="pytorch_model",
        help="Directory for HF saved models. Default is pytorch_model.",
    )
    parser.add_argument("--hf-token", type=str, help="HuggingFace API access token for downloading model checkpoints.")

    # Engine build options.
    parser.add_argument("--force-engine-build", action="store_true", help="Force rebuilding the TensorRT engine.")
    parser.add_argument(
        "--build-dynamic-batch", action="store_true", help="Build TensorRT engines to support dynamic batch size."
    )
    parser.add_argument(
        "--build-dynamic-shape", action="store_true", help="Build TensorRT engines to support dynamic image sizes."
    )

    # Inference related options
    parser.add_argument(
        "--num-warmup-runs", type=int, default=5, help="Number of warmup runs before benchmarking performance."
    )
    parser.add_argument("--nvtx-profile", action="store_true", help="Enable NVTX markers for performance profiling.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator to get consistent results.")
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable cuda graph.")

    parser.add_argument(
        "--disable-refiner", action="store_true", help="Disable refiner and only run base for XL pipeline."
    )

    group = parser.add_argument_group("Options for ORT_CUDA engine only")
    group.add_argument("--enable-vae-slicing", action="store_true", help="True will feed only one image to VAE once.")

    # TensorRT only options
    group = parser.add_argument_group("Options for TensorRT (--engine=TRT) only")
    group.add_argument("--onnx-refit-dir", help="ONNX models to load the weights from.")
    group.add_argument(
        "--build-enable-refit", action="store_true", help="Enable Refit option in TensorRT engines during build."
    )
    group.add_argument(
        "--build-preview-features", action="store_true", help="Build TensorRT engines with preview features."
    )
    group.add_argument(
        "--build-all-tactics", action="store_true", help="Build TensorRT engines using all tactic sources."
    )

    args = parser.parse_args()

    if (
        args.engine in ["ORT_CUDA", "ORT_TRT"]
        and (args.force_onnx_export or args.force_onnx_optimize)
        and not args.force_engine_build
    ):
        raise ValueError(
            "For ORT_CUDA or ORT_TRT, --force_onnx_export and --force_onnx_optimize are not supported. "
            "Please use --force_engine_build instead."
        )

    # Validate image dimensions
    if args.height % 64 != 0 or args.width % 64 != 0:
        raise ValueError(
            f"Image height and width have to be divisible by 64 but specified as: {args.height} and {args.width}."
        )

    if (args.build_dynamic_batch or args.build_dynamic_shape) and not args.disable_cuda_graph:
        print("[I] CUDA Graph is disabled since dynamic input shape is configured.")
        args.disable_cuda_graph = True

    if args.onnx_opset is None:
        args.onnx_opset = 14 if args.engine == "ORT_CUDA" else 17

    if is_xl:
        if args.lcm and args.scheduler != "LCM":
            print("[I] Use --scheduler=LCM for base since LCM is used.")
            args.scheduler = "LCM"
        assert args.strength > 0.0 and args.strength < 1.0
        assert not (args.lcm and args.lora_weights), "it is not supported to use both lcm unet and Lora together"

    if args.scheduler == "LCM":
        if args.guidance > 1.0:
            print("[I] Use --guidance=1.0 for base since LCM is used.")
            args.guidance = 1.0
        if args.denoising_steps > 16:
            print("[I] Use --denoising_steps=8 (no more than 16) for base since LCM is used.")
            args.denoising_steps = 8

    print(args)

    return args


def max_batch(args):
    do_classifier_free_guidance = args.guidance > 1.0
    batch_multiplier = 2 if do_classifier_free_guidance else 1
    max_batch_size = 32 // batch_multiplier
    if args.engine != "ORT_CUDA" and (args.build_dynamic_shape or args.height > 512 or args.width > 512):
        max_batch_size = 8 // batch_multiplier
    return max_batch_size


def get_metadata(args, is_xl: bool = False) -> Dict[str, Any]:
    metadata = {
        "args.prompt": args.prompt,
        "args.negative_prompt": args.negative_prompt,
        "args.batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "cuda_graph": not args.disable_cuda_graph,
        "vae_slicing": args.enable_vae_slicing,
        "engine": args.engine,
    }

    if is_xl and not args.disable_refiner:
        metadata["base.scheduler"] = args.scheduler
        metadata["base.denoising_steps"] = args.denoising_steps
        metadata["base.guidance"] = args.guidance
        metadata["refiner.strength"] = args.strength
        metadata["refiner.scheduler"] = args.refiner_scheduler
        metadata["refiner.denoising_steps"] = args.refiner_steps
        metadata["refiner.guidance"] = args.refiner_guidance
    else:
        metadata["scheduler"] = args.scheduler
        metadata["denoising_steps"] = args.denoising_steps
        metadata["guidance"] = args.guidance

    return metadata


def repeat_prompt(args):
    if not isinstance(args.prompt, list):
        raise ValueError(f"`prompt` must be of type `str` or `str` list, but is {type(args.prompt)}")
    prompt = args.prompt * args.batch_size

    if not isinstance(args.negative_prompt, list):
        raise ValueError(
            f"`--negative-prompt` must be of type `str` or `str` list, but is {type(args.negative_prompt)}"
        )

    if len(args.negative_prompt) == 1:
        negative_prompt = args.negative_prompt * len(prompt)
    else:
        negative_prompt = args.negative_prompt

    return prompt, negative_prompt


def init_pipeline(
    pipeline_class, pipeline_info, engine_type, args, max_batch_size, opt_batch_size, opt_image_height, opt_image_width
):
    onnx_dir, engine_dir, output_dir, framework_model_dir, timing_cache = get_engine_paths(
        work_dir=args.work_dir, pipeline_info=pipeline_info, engine_type=engine_type
    )

    # Initialize demo
    pipeline = pipeline_class(
        pipeline_info,
        scheduler=args.refiner_scheduler if pipeline_info.is_xl_refiner() else args.scheduler,
        output_dir=output_dir,
        hf_token=args.hf_token,
        verbose=False,
        nvtx_profile=args.nvtx_profile,
        max_batch_size=max_batch_size,
        use_cuda_graph=not args.disable_cuda_graph,
        framework_model_dir=framework_model_dir,
        engine_type=engine_type,
    )

    if engine_type == EngineType.ORT_CUDA:
        # Build CUDA EP engines and load pytorch modules
        pipeline.backend.build_engines(
            engine_dir=engine_dir,
            framework_model_dir=framework_model_dir,
            onnx_dir=onnx_dir,
            force_engine_rebuild=args.force_engine_build,
            device_id=torch.cuda.current_device(),
        )
    elif engine_type == EngineType.ORT_TRT:
        # Build TensorRT EP engines and load pytorch modules
        pipeline.backend.build_engines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            args.onnx_opset,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            opt_batch_size=opt_batch_size,
            force_engine_rebuild=args.force_engine_build,
            static_batch=not args.build_dynamic_batch,
            static_image_shape=not args.build_dynamic_shape,
            max_workspace_size=0,
            device_id=torch.cuda.current_device(),
            timing_cache=timing_cache,
        )
    elif engine_type == EngineType.TRT:
        # Load TensorRT engines and pytorch modules
        pipeline.backend.load_engines(
            engine_dir,
            framework_model_dir,
            onnx_dir,
            args.onnx_opset,
            opt_batch_size=opt_batch_size,
            opt_image_height=opt_image_height,
            opt_image_width=opt_image_width,
            force_export=args.force_onnx_export,
            force_optimize=args.force_onnx_optimize,
            force_build=args.force_engine_build,
            static_batch=not args.build_dynamic_batch,
            static_shape=not args.build_dynamic_shape,
            enable_refit=args.build_enable_refit,
            enable_preview=args.build_preview_features,
            enable_all_tactics=args.build_all_tactics,
            timing_cache=timing_cache,
            onnx_refit_dir=args.onnx_refit_dir,
        )

    return pipeline


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")
