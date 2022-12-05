# Attention is all you need: A Pytorch Directml Implementation

This is a PyTorch Directml implementation of the Transformer model in "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017). 


This sample is extracted from [pytorch benchmark](https://github.com/pytorch/benchmark/tree/main/torchbenchmark/models/attention_is_all_you_need_pytorch), and has been slightly changed to apply pytorch-directml.


# Requirement
- python  3.8
- pytorch 1.13
- pytorch_directml 0.1.13
- torchtext 0.14.0
- spacy
- tqdm
- dill
- numpy


# Usage

## Run install.py to download and preprocess data

```ps
python install.py
```

## Train the model
```bash
python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 128 -warmup 128000 -epoch 400 -use_dml
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt -use_dml
```

