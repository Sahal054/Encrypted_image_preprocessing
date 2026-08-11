"""Filter definitions, with pre-processing, post-processing and compilation methods."""

import numpy as np
import torch
from torch import nn
from common import AVAILABLE_FILTERS, INPUT_SHAPE

from concrete.fhe.compilation.compiler import Compiler
from concrete.ml.common.utils import generate_proxy_function
from concrete.ml.torch.numpy_module import NumpyModule


class TorchConv(nn.Module):
    """Torch model with a single convolution operator."""

    def __init__(self, kernel, n_in_channels=3, n_out_channels=3, groups=1, threshold=None):
        """Initialize the filter.

        Args:
            kernel (np.ndarray): The convolution kernel to consider.
            n_in_channels (int): Number of input channels. Default: 3.
            n_out_channels (int): Number of output channels. Default: 3.
            groups (int): Convolution groups. Default: 1.
            threshold (int | None): Value subtracted from the conv output before returning.
        """
        super().__init__()
        self.kernel = torch.tensor(kernel, dtype=torch.int64)
        self.n_out_channels = n_out_channels
        self.n_in_channels = n_in_channels
        self.groups = groups
        self.threshold = threshold

    def forward(self, x):
        """Forward pass with a single convolution using a 1D or 2D kernel.

        Args:
            x (torch.Tensor): The input image.

        Returns:
            torch.Tensor: The filtered image.
        """
        stride = 1
        kernel_shape = self.kernel.shape

        if len(kernel_shape) == 1:
            self.kernel = self.kernel.repeat(self.n_out_channels)
            kernel = self.kernel.reshape(
                self.n_out_channels,
                self.n_in_channels // self.groups,
                1,
                1,
            )
        elif len(kernel_shape) == 2:
            kernel = self.kernel.expand(
                self.n_out_channels,
                self.n_in_channels // self.groups,
                kernel_shape[0],
                kernel_shape[1],
            )
        else:
            raise ValueError(
                "Wrong kernel shape, only 1D or 2D kernels are accepted. Got kernel of shape "
                f"{kernel_shape}"
            )

        # Reshape: (W, H, C) → (B, C, H, W) for torch conv2d
        x = x.transpose(2, 0).unsqueeze(axis=0)

        x = nn.functional.conv2d(x, kernel, stride=stride, groups=self.groups)

        # Reshape back: (B, C_out, H, W) → (H, W, C_out)
        x = x.transpose(1, 3).reshape((x.shape[2], x.shape[3], self.n_out_channels))

        if self.threshold is not None:
            x -= self.threshold

        return x


class Filter:
    """Filter class used in the app.

    Each filter wraps a small TorchConv model whose integer kernel can be
    compiled into an FHE circuit by Concrete-ML.  The available filters are
    deliberately chosen to form a fingerprint-processing pipeline:

        grayscale → blur (noise reduction) → sharpen / ridge-detect →
        sobel / laplacian (feature extraction)
    """

    def __init__(self, filter_name):
        """Initialize the filter class using a given filter.

        Args:
            filter_name (str): The filter to consider.
        """

        assert filter_name in AVAILABLE_FILTERS, (
            f"Unsupported image filter or transformation. Expected one of {*AVAILABLE_FILTERS,}, "
            f"but got {filter_name}",
        )

        self.filter_name = filter_name
        self.onnx_model = None
        self.fhe_circuit = None
        self.divide = None
        self.offset = None          # added to output in post-processing (before clip)

        # ── Preprocessing filters ──────────────────────────────────────────

        if filter_name == "black and white":
            # PAL/NTSC grayscale weights scaled ×1000 for integer FHE
            kernel = [299, 587, 114]
            self.torch_model = TorchConv(kernel)
            self.divide = 1000

        elif filter_name == "blur":
            kernel = np.ones((3, 3))
            self.torch_model = TorchConv(kernel, groups=3)
            self.divide = 9

        # ── Enhancement / ridge analysis filters ───────────────────────────

        elif filter_name == "sharpen":
            kernel = [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0],
            ]
            self.torch_model = TorchConv(kernel, groups=3)

        elif filter_name == "ridge detection":
            kernel = [
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1],
            ]
            self.torch_model = TorchConv(kernel, threshold=900)

        elif filter_name == "fingerprint enhance":
            # Directional ridge enhancer: amplifies horizontal ridge structures
            # while suppressing flat regions.  Combined with a threshold shift
            # so that only ridges above a certain activation survive clipping.
            kernel = [
                [-1, -1, -1],
                [ 2,  2,  2],
                [-1, -1, -1],
            ]
            self.torch_model = TorchConv(kernel, groups=3, threshold=300)

        # ── Directional feature-extraction filters ─────────────────────────

        elif filter_name == "sobel horizontal":
            # Detects horizontal edges (vertical ridges in a fingerprint)
            kernel = [
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1],
            ]
            self.torch_model = TorchConv(kernel, groups=3)

        elif filter_name == "sobel vertical":
            # Detects vertical edges (horizontal ridges in a fingerprint)
            kernel = [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1],
            ]
            self.torch_model = TorchConv(kernel, groups=3)

        elif filter_name == "laplacian":
            # Second-derivative edge detector — direction-independent ridges
            kernel = [
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0],
            ]
            self.torch_model = TorchConv(kernel, groups=3)

    # ------------------------------------------------------------------ compile

    def compile(self):
        """Compile the filter on a representative inputset."""
        np.random.seed(42)
        inputset = tuple(
            np.random.randint(0, 256, size=(INPUT_SHAPE + (3,)), dtype=np.int64)
            for _ in range(100)
        )

        numpy_module = NumpyModule(
            self.torch_model,
            dummy_input=torch.from_numpy(inputset[0]),
        )

        numpy_filter_proxy, parameters_mapping = generate_proxy_function(
            numpy_module.numpy_forward,
            ["inputs"],
        )

        compiler = Compiler(
            numpy_filter_proxy,
            {parameters_mapping["inputs"]: "encrypted"},
        )
        self.fhe_circuit = compiler.compile(inputset)

        return self.fhe_circuit

    # -------------------------------------------------------------- post-process

    def post_processing(self, output_image):
        """Apply post-processing to the decrypted output image.

        Args:
            output_image (np.ndarray): The decrypted image.

        Returns:
            np.ndarray: The post-processed image.
        """
        if self.divide is not None:
            output_image //= self.divide

        if self.offset is not None:
            output_image = output_image + self.offset

        output_image = output_image.clip(0, 255)

        return output_image