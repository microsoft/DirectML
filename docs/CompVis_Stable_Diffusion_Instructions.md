# Running CompVis Stable Diffusion on a Single GPU with ONNX Runtime and DirectML

These instructions download and set up the CompVis Stable Diffusion v1.4 model through the Hugging Face diffusers and transformers library. It pulls relevant Python packages that allow the model to run on most discrete consumer graphics GPUs with ONNX Runtime atop the DirectML execution provider. These instructions are based on the prior work of [Neil McAlister](https://www.travelneil.com/stable-diffusion-windows-amd.html) with the more up-to-date script version from the Hugging Face Diffusers repo and its dependency packages, as well as additional conversion steps for better execution performance.

## Overview

The figure below provides an overview of the components involved in the model conversion process:

![Stable Diffusion Conversion](sd_conversion.png)

- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) : the original stable diffusion model source
- [Hugging Face](https://huggingface.co/) : provides APIs, packages, tools, and pretrained models for a wide variety of scenarios including stable diffusion
  - (Python package) [diffusers](https://github.com/huggingface/diffusers) : offers APIs/scripts for diffusion models
  - (Python package) [transformers](https://github.com/huggingface/transformers) : offers APIs/scripts for transformer models
  - (Python script) [convert_stable_diffusion_checkpoint_to_onnx.py](https://github.com/HuggingFace/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py) : converts PyTorch implementation of stable diffusion to ONNX models
- (Python packages) PyTorch ([torch](https://pypi.org/project/torch/) and [torchvision](https://pypi.org/project/torchvision/), and [torchaudio](https://pypi.org/project/torchaudio/)) : the ML framework used for stable diffusion models
- (Python package) [onnx](https://pypi.org/project/onnx/) : for creating ONNX representations of ML models
- (Python package) [onnxruntime-directml](https://pypi.org/project/onnxruntime-directml/) : for running ONNX models with DirectML support

## Installing Dependency Packages

We need a few Python packages, namely the Hugging Face script libraries for transformers and diffusers along with ONNX Runtime for DirectML.

```
pip install diffusers transformers onnxruntime-directml onnx
```

You will also need PyTorch installed to run the Hugging Face model conversion script (`convert_stable_diffusion_checkpoint_to_onnx.py`). By default, the conversion script will output FP32 ONNX models. FP16 models consume less memory and may be faster depending on your GPU. However, **as of this writing, the Hugging Face stable diffusion to ONNX conversion script only supports FP16 if you have PyTorch with CUDA**. This will require up to 3 GB of additional disk space.

**If you have a CUDA-capable graphics card**:
```
pip install torch==1.13.1+cu116 torchaudio==0.13.1+cu116 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116
```

**If you do not have a CUDA-capable graphics card**::
```
pip install torch==1.13.1 torchaudio==0.13.1 torchvision==0.14.1
```

**⚠️ WARNING ⚠️** : Conversion to ONNX with PyTorch 2.0 does not currently work. If you encounter the following error, please make sure that you are using PyTorch 1.13.1 or older. See https://github.com/pytorch/pytorch/issues/96944.

    aten::scaled_dot_product_attention' to ONNX opset version 14 is not supported

### Hardware Requirement
Since the entire model must fit within GPU memory while executing, the GPU should have at least 8GB of VRAM available to run this model. Here are a few examples:
- NVIDIA GeForce RTX 2070 or later
- AMD Radeon RX 6500 XT (8GB) or later
- Intel Arc A750 Graphics or later 

## Downloading the Model
We first need to download the model from [Hugging Face](https://huggingface.co/), for which you need an account. So if you haven't created one, now is the time. Once you've set up a Hugging Face account, generate an [access token](https://huggingface.co/docs/hub/security-tokens) (just follow their instructions in the web site).

Once you have the account and an access token, authenticate yourself in a terminal or powershell console by running the following command.

```
huggingface-cli.exe login
```

It'll ask for your access token, which you can find on your account profile `Settings -> Access Tokens`, just copy it from here and carefully paste it on this prompt. Note that you won't see anything appear on the prompt when you paste it, that's fine. It's there already, just hit Enter. You'll start downloading the model from Hugging Face.

## Converting to ONNX

The model is trained with PyTorch so it can naturally convert to ONNX. Since we'll be using DirectML through ONNX Runtime, this step is needed. The script `convert_stable_diffusion_checkpoint_to_onnx.py`, which you will use here is just a local copy of the same file from the [Hugging Face diffusers GitHub repo](https://github.com/HuggingFace/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py). In case you don't want to clone that entire repo, just copy the file over.

```
python convert_stable_diffusion_checkpoint_to_onnx.py --model_path="CompVis/stable-diffusion-v1-4" --output_path="./stable_diffusion_onnx" --fp16
```

This will run the conversion and put the result ONNX files under the `stable_diffusion_onnx` folder. For better performance, we recommend you convert the model to half-precision floating point data type using the `--fp16` option (as mentioned earlier, you must have PyTorch with CUDA support to use `--fp16`).

## Running the ONNX Model

You'll need a script that looks like what follows. On an NVIDIA GeForce RTX 2070, a single image currently takes about 20 seconds to generate from a prompt. It'll take between 5-10 mins on a CPU.

```python
# (test/run.py)
from diffusers import StableDiffusionOnnxPipeline
pipe = StableDiffusionOnnxPipeline.from_pretrained("./stable_diffusion_onnx", provider="DmlExecutionProvider")
prompt = "a photo of an astronaut riding a horse on mars."
image = pipe(prompt).images[0]
image.save("./result.png")
```

### A Debugging Note
When running this script inside VSCode, the relative path specified here is relative to the base location of your project folder and not the location of your script file. To fix that up, configure the `cwd` (i.e. "current working directory") option in your launch.json file as follows:

```json
    // .vscode/launch.json
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceFolder}/test/",
            "console": "integratedTerminal",
            "justMyCode": true
        }
    ]
```

If you have an NVIDIA graphics card and want to try running the ONNX model on CUDA, just replace the `onnxruntime-directml` package with the `onnxruntime-gpu` package. Do not keep them both. Then replace the `"DmlExecutionProvider"` name in the running script `run.py` with `"CUDAExecutionProvider"`. You may need to install NVIDIA CUDA libraries separately.
