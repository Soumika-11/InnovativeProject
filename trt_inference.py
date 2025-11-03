"""trt_inference.py
Helpers to load a TensorRT engine and run inference on Jetson (or any machine
with TensorRT + PyCUDA installed).

Usage summary:
  from trt_inference import TRTInfer
  infer = TRTInfer(engine_path='model.trt')
  emb = infer.predict(image_array)  # image_array is HxWxC (RGB) float32 in [0,1]

Notes:
- This module expects a TensorRT engine built from an ONNX model that accepts
  a single image input and returns a 1-D embedding (or similar).
- On Jetson, build the engine with `trtexec --onnx=model.onnx --saveEngine=model.trt --fp16`
  (see README_Jetson.md for full steps).

IMPORTANT - Platform Requirements:
- TensorRT is NOT available on macOS (Linux/Windows only, requires NVIDIA GPU)
- On macOS: This module can be imported but TRTInfer() will raise RuntimeError
- On Jetson Nano: TensorRT comes pre-installed with JetPack SDK (do NOT pip install)
- For development on macOS, use TensorFlow/Keras models (.h5) instead

Requires: tensorrt (system-provided on Jetson), pycuda, numpy, opencv-python.
"""

import os
import numpy as np
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 (initializes CUDA driver)
except Exception as e:
    # We don't raise here so the module can be imported on non-Jetson hosts for editing
    trt = None
    cuda = None
    _import_error = e

import cv2


class TRTInfer:
    """Simple wrapper to load TensorRT engine and run synchronous inference.

    Constructor arguments:
        engine_path: path to .trt engine file (binary)
        input_shape: tuple (H, W, C) or None. If None, will use engine binding shape.
        dtype: numpy dtype for input (default np.float32)
    """

    def __init__(self, engine_path: str, input_shape=None, dtype=np.float32):
        if trt is None:
            raise RuntimeError(f"TensorRT / PyCUDA import failed: {_import_error}")
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")

        self.engine_path = engine_path
        self.dtype = dtype
        self.TRLogger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f, trt.Runtime(self.TRLogger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError('Failed to deserialize engine')

        self.context = self.engine.create_execution_context()
        # Allocate buffers
        self.bindings = []
        self.binding_addrs = {}
        self.host_buffers = {}
        self.device_buffers = {}
        self.stream = cuda.Stream()

        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            self.bindings.append(name)
            is_input = self.engine.binding_is_input(idx)
            shape = self.engine.get_binding_shape(idx)
            # If dynamic shape (-1), we will need to set shapes before inference
            self.host_buffers[name] = None
            self.device_buffers[name] = None

        # Try to detect input binding name and shape
        self.input_binding = None
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx):
                self.input_binding = self.engine.get_binding_name(idx)
                try:
                    bs = tuple(self.engine.get_binding_shape(idx))
                except Exception:
                    bs = None
                break

        if input_shape is not None:
            self.input_shape = input_shape
        elif bs is not None:
            # binding shape may be like (C,H,W) or (N,C,H,W) depending on engine
            self.input_shape = self._normalize_binding_shape(bs)
        else:
            self.input_shape = None

    def _normalize_binding_shape(self, binding_shape):
        # Accept common patterns: (N,C,H,W) or (C,H,W) or (-1, C, H, W)
        arr = list(binding_shape)
        if len(arr) == 4:
            # assume NCHW -> drop batch dim
            _, c, h, w = arr
            return (h, w, c)
        if len(arr) == 3:
            # CHW
            c, h, w = arr
            return (h, w, c)
        # fallback
        return None

    def _prepare_input(self, image: np.ndarray):
        """Accepts an HxWxC RGB float32 image in [0,1] and resizes/normalizes it
        to the engine input shape (if known). Returns a contiguous numpy array
        ready for copying to the host buffer.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        if self.input_shape is not None:
            h, w, c = self.input_shape
            if image.shape[:2] != (h, w):
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # Many TensorRT engines expect NCHW; we'll leave layout conversion to the engine
        # Here we assume the ONNX model was exported with input layout matching what the engine expects.
        # Return a batch dimension added array
        batched = np.expand_dims(image, axis=0)
        return np.ascontiguousarray(batched)

    def _allocate_binding_memory(self, name, shape, dtype=np.float32):
        # shape is a tuple for the host buffer
        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, np.float32)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        self.host_buffers[name] = host_mem
        self.device_buffers[name] = device_mem
        return host_mem, device_mem

    def predict(self, image: np.ndarray):
        """Run inference on a single image (HxWxC, RGB, float32 normalized [0,1]).
        Returns the numpy output array(s) (flattened if single output).
        """
        # Prepare input
        inp = self._prepare_input(image)
        # Determine input binding shape
        # We assume single input and single output engine for the feature extractor
        if self.input_binding is None:
            raise RuntimeError('No input binding detected in engine')

        # Set execution context shape if dynamic
        binding_idx = self.engine.get_binding_index(self.input_binding)
        binding_shape = self.engine.get_binding_shape(binding_idx)
        # If dynamic (contains -1) set binding shape using context.set_binding_shape
        if any([d == -1 for d in binding_shape]):
            # Try to set shape as NCHW if context expects that
            # We need to get shape as (N,C,H,W)
            # Here we attempt a common conversion: assume engine expects (N,C,H,W)
            n = inp.shape[0]
            h, w, c = inp.shape[1:]
            new_shape = (n, c, h, w)
            self.context.set_binding_shape(binding_idx, new_shape)

        # Allocate host/device memory for all bindings on first run
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            is_input = self.engine.binding_is_input(idx)
            shape = self.context.get_binding_shape(idx)
            # Convert shape to tuple of ints
            shape = tuple(int(x) for x in shape)
            if self.host_buffers.get(name) is None:
                self._allocate_binding_memory(name, shape)

        # Copy input into host buffer (flatten)
        host_in = self.host_buffers[self.input_binding]
        np.copyto(host_in, inp.ravel())

        # Bindings array (device addresses) in order expected by engine
        bindings = [int(self.device_buffers[self.engine.get_binding_name(i)]) for i in range(self.engine.num_bindings)]

        # Copy host to device for inputs
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if self.engine.binding_is_input(idx):
                cuda.memcpy_htod_async(int(self.device_buffers[name]), np.ascontiguousarray(self.host_buffers[name]), self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Copy outputs back to host
        outputs = {}
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            if not self.engine.binding_is_input(idx):
                host_out = self.host_buffers[name]
                cuda.memcpy_dtoh_async(host_out, int(self.device_buffers[name]), self.stream)
                outputs[name] = np.array(host_out).copy()

        # Synchronize stream
        self.stream.synchronize()

        # If single output, return flattened vector
        if len(outputs) == 1:
            out = list(outputs.values())[0]
            return out.reshape(-1)
        return outputs


if __name__ == '__main__':
    print('trt_inference module: helper for loading .trt engines and running inference')
