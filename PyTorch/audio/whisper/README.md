# Speech Recognition with Whisper
This sample guides you on how to run OpenAI's automatic speech recognition (ASR) [Whisper model](https://github.com/openai/whisper/blob/main/README.md) with our DirectML-backend.

- [Setup](#setup)
- [About Whisper](#run-the-whisper-model)
- [Basic Settings](#basic-settings)
- [External Links](#external-links)
- [Model License](#model-license)


## About Whisper 

The [OpenAI Whisper](https://github.com/openai/whisper/) model is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification. 

Whisper supports five model sizes, four with English-only versions and all five with multilingual versions. 
|  Size     | Parameters | English-only model | Multilingual model | Required VRAM 
|:---------:|:----------:|:------------------:|:------------------:|:-------------:|
|  tiny     |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |
|  base     |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |
| small     |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |
| medium    |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |
| large v3  |   1550 M   |        N/A         |      `large-v3`    |    ~10 GB     |

For more information on the model, please refer to the [OpenAI Whisper GitHub repo](https://github.com/openai/whisper/).


## Setup
Once you've setup `torch-directml` following our [Windows](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows) and [WSL](https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-wsl) guidance, install the following requirements for running the app:


```
conda install ffmpeg
pip install -r requirements.txt
```


## Run the Whisper model
Run Whisper with DirectML backend with a sample audio file with the following command: 
```bash
python run.py --input_file <audio_file> --model_size "tiny.en"
```


For example, you should see the result output as below:  
```
> python run.py --input_file test/samples_jfk.wav --model_size "tiny.en"
100%|█████████████████████████████████████| 72.1M/72.1M [00:09<00:00, 7.90MiB/s]
test/samples_jfk.wav

And so my fellow Americans ask not what your country can do for you ask what you can do for your country.
```


Note, by default [OpenAI Whisper](https://github.com/openai/whisper/) uses a naive implementation for the scaled dot product attention. If you want to improve performance further to leverage DirectML's scaled dot product attention, execute `run.py` with `--use_dml_attn` flag: 

```bash
python run.py --input_file <audio_file> --model_size "tiny.en" --use_dml_attn
```
Based on this flag `MultiHeadAttention` module in `model.py` would choose between naive whisper scaled dot product attention and DirectML's scaled dot product attention:
```python
if use_dml_attn:
    wv, qk = self.dml_sdp_attn(q, k, v, mask, cross_attention=cross_attention)
else:
    wv, qk = self.qkv_attention(q, k, v, mask)
```

## Basic Settings

Following is a list of the basic settings supported by `run.py`:



| Flag                   | Description                                                  | Default |
| ---------------------- | ------------------------------------------------------------ | ------- |
| `--help`            | Show this help message. | - |
| `--input_file`          | [Required] Path to input audio file  | - |
| `--model_size`         | Size of Whisper model to use.   Options: [`tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v3`]           | `tiny.en` |
| `--fp16`    | Runs inference with fp16 precision. | True |
| `--use_dml_attn`    | Runs inference with DirectML Scaled dot product attention impl. | False |


## External Links
- [Whisper Base Hugging Face Repository](https://huggingface.co/openai/whisper-base.en)   
- [Whisper Tiny Hugging Face Repository](https://huggingface.co/openai/whisper-tiny.en)  
- [Whisper Small Hugging Face Repository](https://huggingface.co/openai/whisper-small.en)  
- [Whisper Medium Hugging Face Repository](https://huggingface.co/openai/whisper-medium.en)  
- [Whisper Large v3 Hugging Face Repository](https://huggingface.co/openai/whisper-large-v3)  
- [Whisper GitHub Repo](https://github.com/openai/whisper)



## Model License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.