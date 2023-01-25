import os
import sys
import subprocess
from pathlib import Path

def pip_install_requirements():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', os.path.join(current_dir, 'requirements.txt')])

def spacy_download(language):
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', language])

def preprocess():
    current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    data_root = os.path.join(current_dir, ".data")
    os.makedirs(data_root, exist_ok=True)
    subprocess.check_call([sys.executable, os.path.join(current_dir, 'preprocess.py'), '-lang_src', 'de', '-lang_trg', 'en', '-share_vocab',
                           '-save_data', os.path.join(data_root, 'm30k_deen_shr.pkl')])

if __name__ == '__main__':
    pip_install_requirements()
    spacy_download('en')
    spacy_download('de')
    # Preprocessed pkl is larger than 100MB so we cannot skip preprocess
    preprocess()
