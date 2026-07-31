# Audio2Face in a browser: feasibility note

As of July 2026, browser inference is plausible as an experimental research
track, but it is not a drop-in production path.

NVIDIA now publishes Audio2Face-3D open weights in ONNX form and an MIT-licensed
SDK. The smaller Mark v2.3 regression release has an approximately 76 MB ONNX
network, but its complete downloadable model is roughly 320 MB before browser
compression and includes model data plus blendshape/post-processing assets.
The v3 diffusion network alone is roughly 725 MB. NVIDIA's supported runtime is
its C++ Audio2Face SDK using CUDA and TensorRT, not ONNX Runtime Web.

The practical paths are:

1. **Production now:** run the official SDK or NIM on a scale-to-zero GPU and
   stream its blendshape frames to the browser. This maps cleanly onto app.nz
   Cog/RunPod infrastructure.
2. **Browser experiment:** test Mark v2.3 with ONNX Runtime Web/WebGPU, first
   validating operator coverage, stateful 8,320-sample audio windows, the
   4,160-sample hop, and NVIDIA's blendshape solver/post-processing. Ship it only
   after measuring download, memory, realtime factor, and output parity on
   Chrome, Safari, and mobile.
3. **Small fallback:** retain the existing local viseme/envelope driver for
   instant startup and privacy-sensitive/offline use.

Do not simply load `network.onnx` and call the feature complete: the SDK also
handles buffering, emotion state, temporal smoothing, jaw/eye transforms, and
conversion of model geometry output to a target character's blendshape set.

Primary references:

- https://github.com/NVIDIA/Audio2Face-3D
- https://github.com/NVIDIA/Audio2Face-3D-SDK
- https://huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark
- https://huggingface.co/nvidia/Audio2Face-3D-v3.0

