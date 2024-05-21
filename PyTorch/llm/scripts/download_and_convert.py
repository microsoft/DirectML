# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from requests.exceptions import HTTPError

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from models.phi2 import ModelArgs as Phi2ModelArgs
from models.phi3 import ModelArgs as Phi3ModelArgs
from models.llama import ModelArgs as LlamaModelArgs

def hf_download(model_repo: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    checkpoint_dir = f"checkpoints/{model_repo}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.listdir(checkpoint_dir):
        print(f"The directory {checkpoint_dir} is not empty. Skipping download.")
    else:
        try:
            snapshot_download(model_repo, local_dir=checkpoint_dir, local_dir_use_symlinks=False, token=hf_token)
        except HTTPError as e:
            if e.response.status_code == 401:
                print("You need to pass a valid `--hf_token=...` to download private checkpoints.")
            else:
                raise e
    return checkpoint_dir

@torch.inference_mode()
def convert_hf_checkpoint(
    *,
    checkpoint_dir: Path = Path("microsoft/Phi-3-mini-4k-instruct"),
    weight_map_path: str = "config/weight_map.json",
    model_name: Optional[str] = None,
) -> None:
    if os.path.exists(checkpoint_dir / "model.pth"):
        print(f"Converted checkpoint already exists here {checkpoint_dir / 'model.pth'}. Skipping Conversion.")
        return

    with open(weight_map_path, 'r') as file:
        weight_maps = json.load(file)
        
    if model_name is None:
        model_name = checkpoint_dir.name

    is_llama3 = "Llama-3" in model_name
    is_phi3 = "Phi-3" in model_name

    if "phi" not in model_name.lower():
        weight_map = weight_maps["llama"]
        config = LlamaModelArgs.from_name(model_name)
    else:
        weight_map = weight_maps[model_name]
        if is_phi3:
            config = Phi3ModelArgs.from_name(model_name)
        else:
            config = Phi2ModelArgs.from_name(model_name)
    
    # Load the json file containing weight mapping
    model_map_json = checkpoint_dir / "model.safetensors.index.json"

    assert model_map_json.is_file()

    with open(model_map_json) as json_map:
        bin_index = json.load(json_map)

    bin_files = {checkpoint_dir / bin for bin in bin_index["weight_map"].values()}
    
    merged_result = {}
    for file in sorted(bin_files):
        state_dict = load_file(str(file))
        merged_result.update(state_dict)
    
    final_result = {}
    for key, value in merged_result.items():
        if "layers" in key:
            abstract_key = re.sub(r'(\d+)', '{}', key, count=1)
            layer_num = re.search(r'\d+', key).group(0)
            new_key = weight_map[abstract_key]
            if new_key == "":
                continue
            new_key = new_key.format(layer_num)
        else:
            new_key = weight_map[key]

        final_result[new_key] = value

    for key in tuple(final_result.keys()):
        if "wq.weight" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=0)

            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

        if "wq.bias" in key:
            q = final_result[key]
            k = final_result[key.replace("wq", "wk")]
            v = final_result[key.replace("wq", "wv")]
            final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v], dim=0)

            del final_result[key]
            del final_result[key.replace("wq", "wk")]
            del final_result[key.replace("wq", "wv")]

    print(f"Saving checkpoint to {checkpoint_dir / 'model.pth'}")
    torch.save(final_result, checkpoint_dir / "model.pth")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download and convert HuggingFace checkpoint.')
    parser.add_argument('--model_repo', type=str, default="microsoft/Phi-3-mini-4k-instruct", help='Huggingface Repository ID to download from.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')
    parser.add_argument('--model_name', type=str, default=None)

    args = parser.parse_args()
    checkpoint_dir = hf_download(args.model_repo, args.hf_token)
    convert_hf_checkpoint(
        checkpoint_dir=Path(checkpoint_dir),
        model_name=args.model_name,
    )
