# Copyright (c) Microsoft Corporation.
# 
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import whisper
import torch_directml
import argparse


def main(args):
    device = torch_directml.device(torch_directml.default_device())
    model = whisper.load_model(args.model_size, device=device, use_dml_attn=args.use_dml_attn)

    # Load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(args.input_file)
    audio = whisper.pad_or_trim(audio)
    
    n_mels = 80
    if args.model_size == "large-v3":
        n_mels = 128

    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).to(model.device)
    language = "en"
    if "en" not in args.model_size:
        _, probs = model.detect_language(mel)
        language = max(probs, key=probs.get)
        print(f"Detected language: {language}")

    options = whisper.DecodingOptions(language=language, fp16=args.fp16)
    result = whisper.decode(model, mel, options)
    
    print(result.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Whisper model on specified audio file with warmup.')
    parser.add_argument('--model_size', type=str, default='tiny.en', help='Size of the Whisper model to use.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input audio file.')
    parser.add_argument('--fp16', action="store_true", help='Runs inference with fp16 precision.')
    parser.add_argument('--use_dml_attn', action="store_true", help='Use DirectML attention implementation.')
    args = parser.parse_args()

    main(args)
