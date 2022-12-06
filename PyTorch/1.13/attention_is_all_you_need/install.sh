pip install -r requirements.txt
python -m spacy download en
python -m spacy download de
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
