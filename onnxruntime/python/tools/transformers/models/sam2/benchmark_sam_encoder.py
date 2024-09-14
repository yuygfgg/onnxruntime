# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Benchmark performance of SAM2 encoder with ORT or PyTorch.

curl https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt > sam2_hiera_base_plus.pt

"""

import argparse
import csv
import statistics
import time
from datetime import datetime
from typing import List, Mapping, Optional

import torch
from convert_to_onnx import SAM2ImageEncoder

from onnxruntime import InferenceSession, SessionOptions, get_available_providers
from onnxruntime.transformers.io_binding_helper import CudaSession, ShapeDict


class TestConfig:
    def __init__(
        self,
        model_name: str,
        onnx_path: str,
        checkpoint_path: str,
        batch_size: int,
        height: int,
        width: int,
        provider="CPUExecutionProvider",
        device: Optional[torch.device] = None,
        use_tf32: bool = True,
        enable_cuda_graph: bool = False,
        dtype=torch.float,
        repeats: int = 100,
        verbose: bool = False,
    ):
        assert model_name in ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"]
        assert height >= 160 and height <= 4096 and height % 16 == 0
        assert width >= 160 and width <= 4096 and width % 16 == 0
        assert checkpoint_path.endswith(f"{model_name}.pt")

        self.model_name = model_name
        self.onnx_path = onnx_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.provider = provider
        self.device = device
        self.use_tf32 = use_tf32
        self.enable_cuda_graph = enable_cuda_graph
        self.dtype = dtype
        self.repeats = repeats
        self.verbose = verbose

    def __repr__(self):
        return (
            f"TestConfig(batch_size={self.batch_size}, height={self.height}, width={self.width}, "
            f"provider={self.provider}, device={self.device}, enable_cuda_graph={self.enable_cuda_graph}, "
            f"dtype={self.dtype})"
        )

    def model_cfg(self):
        if self.model_name == "sam2_hiera_tiny":
            model_cfg = "sam2_hiera_t.yaml"
        elif self.model_name == "sam2_hiera_small":
            model_cfg = "sam2_hiera_s.yaml"
        elif self.model_name == "sam2_hiera_base_plus":
            model_cfg = "sam2_hiera_b+.yaml"
        else:
            model_cfg = "sam2_hiera_l.yaml"
        return model_cfg

    def shape_dict(self) -> ShapeDict:
        return {
            "image": [self.batch_size, 3, self.height, self.width],
            "high_res_feats_0": [self.batch_size, 32, self.height // 4, self.width // 4],
            "high_res_feats_1": [self.batch_size, 64, self.height // 8, self.width // 8],
            "image_embeddings": [self.batch_size, 256, self.height // 16, self.width // 16],
        }

    def symbolic_shape_dict(self) -> Mapping[str, List[int | str]]:
        return {
            "image": ["batch_size", 3, "height", "width"],
            "image_embeddings": ["batch_size", 256, "height/16", "width/16"],
            "high_res_feats_0": ["batch_size", 32, "height/4", "width/4"],
            "high_res_feats_1": ["batch_size", 64, "height/8", "width/8"],
        }

    def random_inputs(self, seed: int = 123, no_bias_k_v: bool = False):
        image = (torch.rand(self.batch_size, 3, self.height, self.width) * 255).to(torch.uint8)
        return {
            "image": image.to(device=self.device, dtype=self.dtype),
        }


def create_ort_session(config: TestConfig, session_options=None) -> InferenceSession:
    if config.verbose:
        print(f"create session for {vars(config)}")

    if config.provider == "CUDAExecutionProvider":
        device_id = torch.cuda.current_device() if isinstance(config.device, str) else config.device.index
        provider_options = CudaSession.get_cuda_provider_options(device_id, config.enable_cuda_graph)
        provider_options["use_tf32"] = int(config.use_tf32)
        provider_options["prefer_nhwc"] = 1
        providers = [(config.provider, provider_options), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    ort_session = InferenceSession(config.onnx_path, session_options, providers=providers)
    return ort_session


def create_session(config: TestConfig, session_options=None) -> CudaSession:
    ort_session = create_ort_session(config, session_options)
    cuda_session = CudaSession(ort_session, config.device, config.enable_cuda_graph)
    shape_dict = config.shape_dict()
    cuda_session.allocate_buffers(shape_dict)
    return cuda_session


class OrtTestSession:
    """A wrapper of ORT session to test relevance and performance."""

    def __init__(self, config: TestConfig, session_options=None):
        self.ort_session = create_session(config, session_options)
        self.feed_dict = config.random_inputs()

    def infer(self):
        return self.ort_session.infer(self.feed_dict)


def measure_latency(cuda_session: CudaSession, input_dict):
    start = time.time()
    _ = cuda_session.infer(input_dict)
    end = time.time()
    return end - start


def run_torch(
    config: TestConfig, compile_mode="max-autotune"
):  # or "reduce-overhead" if you want to reduce latency of first run.
    model_cfg = config.model_cfg()
    sam2_checkpoint = config.checkpoint_path

    device_type = config.device.type
    is_cuda = device_type == "cuda"
    # Turn on TF32 for Ampere GPUs which could help when data type is float32.
    if is_cuda and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=config.dtype):
        from sam2.build_sam import build_sam2

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=config.device)
        sam2_model.image_encoder.forward = torch.compile(
            sam2_model.image_encoder.forward,
            mode=compile_mode,
            fullgraph=True,
            dynamic=False,
        )

        image_shape = config.shape_dict()["image"]
        img = torch.randn(image_shape).to(device=config.device, dtype=config.dtype)
        sam2_encoder = SAM2ImageEncoder(sam2_model)

        # warm up
        for _ in range(3):
            high_res_feats_0, high_res_feats_1, image_embeddings = sam2_encoder(img)
            print("high_res_feats_0 shape:", high_res_feats_0.shape)
            print("high_res_feats_1 shape:", high_res_feats_1.shape)
            print("image_embeddings shape:", image_embeddings.shape)

        start = time.time()
        for _ in range(config.repeats):
            high_res_feats_0, high_res_feats_1, image_embeddings = sam2_encoder(img)
            if is_cuda:
                torch.cuda.synchronize()
        end = time.time()

        return (end - start) / config.repeats


def run_test(
    csv_writer: csv.DictWriter,
    args: argparse.Namespace,
):
    use_gpu: bool = args.use_gpu
    enable_cuda_graph: bool = args.use_cuda_graph
    repeats: int = args.repeats

    if use_gpu:
        device_id = torch.cuda.current_device()
        device = torch.device("cuda", device_id)
        provider = "CUDAExecutionProvider"
    else:
        device_id = 0
        device = torch.device("cpu")
        enable_cuda_graph = False
        provider = "CPUExecutionProvider"

    print("\nmodel\tbatch\theight\twidth\tthreads\tlatency\tengine")

    config = TestConfig(
        model_name=args.model_name,
        onnx_path=args.onnx_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        provider=provider,
        device=device,
        use_tf32=True,
        enable_cuda_graph=False,
        dtype=torch.float,  # TODO: torch.float16
        repeats=args.repeats,
        verbose=False,
    )

    if args.engine == "ort":
        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = args.intra_op_num_threads

        session = create_session(config, sess_options)
        input_dict = config.random_inputs()

        # warm up session
        try:
            _ = measure_latency(session, input_dict)
        except Exception as e:
            print(f"Failed to run {config=}. Exception: {e}")
            return

        latency_list = []
        for _ in range(repeats):
            latency = measure_latency(session, input_dict)
            latency_list.append(latency)
        average_latency = statistics.mean(latency_list)

        del session
    else:  # torch
        with torch.no_grad():
            try:
                average_latency = run_torch(config)
            except Exception as e:
                print(f"Failed to run {config=}. Exception: {e}")
                return

    engine = args.engine + ":" + ("cuda" if use_gpu else "cpu")
    row = {
        "model_name": args.model_name,
        "use_gpu": use_gpu,
        "enable_cuda_graph": enable_cuda_graph,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "intra_op_num_threads": args.intra_op_num_threads,
        "engine": engine,
        "average_latency": average_latency,
    }
    csv_writer.writerow(row)

    print(f"{args.batch_size}\t{args.height}\t{args.width}\t{args.intra_op_num_threads}\t{average_latency}\t{engine}")


def run_tests(args):
    features = "gpu" if args.use_gpu else "cpu"
    csv_filename = "benchmark_sam_{}_{}_{}.csv".format(
        features,
        args.engine,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    with open(csv_filename, mode="a", newline="") as csv_file:
        column_names = [
            "model_name",
            "use_gpu",
            "enable_cuda_graph",
            "batch_size",
            "height",
            "width",
            "intra_op_num_threads",
            "average_latency",
            "engine",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
        csv_writer.writeheader()

        run_test(csv_writer, args)


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark SMA2 for ONNX Runtime and PyTorch.")

    parser.add_argument(
        "--use_gpu",
        required=False,
        action="store_true",
        help="Use GPU for inference.",
    )
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "--use_cuda_graph",
        required=False,
        action="store_true",
        help="Use cuda graph in onnxruntime.",
    )
    parser.set_defaults(use_cuda_graph=False)

    parser.add_argument(
        "--intra_op_num_threads",
        required=False,
        type=int,
        choices=[0, 1, 2, 4, 8, 16],
        default=0,
        help="intra_op_num_threads for onnxruntime. ",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        required=False,
        type=int,
        default=1,
        help="batch size",
    )

    parser.add_argument(
        "-t",
        "--height",
        required=False,
        type=int,
        default=1024,
        help="image height",
    )

    parser.add_argument(
        "-w",
        "--width",
        required=False,
        type=int,
        default=1024,
        help="image width",
    )

    parser.add_argument(
        "-r",
        "--repeats",
        required=False,
        type=int,
        default=0,
        help="number of repeats for performance test",
    )

    parser.add_argument(
        "--engine",
        required=False,
        type=str,
        default="ort",
        choices=["ort", "torch"],
        help="engine for inference",
    )

    parser.add_argument(
        "--model_name",
        required=False,
        type=str,
        default="sam2_hiera_base_plus",
        choices=["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"],
        help="sam2 model name",
    )

    parser.add_argument(
        "--checkpoint_path",
        required=False,
        type=str,
        default="sam2_hiera_base_plus.pt",
        help="sam2 torch model checkpoint path like `checkpoints/sam2_hiera_base_plus.pt`",
    )

    parser.add_argument(
        "--onnx_path",
        required=False,
        type=str,
        default="sam2_hiera_base_plus.onnx",
        help="path of onnx model",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_arguments()

    if args.repeats == 0:
        args.repeats = 1000 if args.use_gpu else 100

    print(f"arguments:{args}")

    if args.use_gpu:
        assert torch.cuda.is_available()
        if args.engine == "ort":
            assert "CUDAExecutionProvider" in get_available_providers()

    run_tests(args)
