"""
Script de téléchargement du dataset MovieLens 32M.
Usage : python data/download_dataset.py
"""
import urllib.request
import zipfile
import os

URL      = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ZIP_PATH = "ml-32m.zip"
OUT_DIR  = "ml-32m"

if os.path.exists(OUT_DIR):
    print(f"Dataset deja present dans '{OUT_DIR}/'")
else:
    print("Telechargement du dataset MovieLens 32M (~300 Mo)...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Extraction...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(".")
    os.remove(ZIP_PATH)
    print(f"Done. Fichiers disponibles dans '{OUT_DIR}/'")
