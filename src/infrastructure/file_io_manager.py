# infrastructure/file_io_manager.py
import os
import json
import pandas as pd
import joblib
import torch

class FileIOManager:
    """
    Gère les opérations d'entrée/sortie : lecture et écriture de fichiers CSV, Excel, JSON, Pickle et Torch.
    """
    def read_csv(self, filepath, **kwargs):
        return pd.read_csv(filepath, **kwargs)

    def read_excel(self, filepath, **kwargs):
        return pd.read_excel(filepath, **kwargs)

    def read_json(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, data, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def list_files(self, folder, extension=None):
        files = os.listdir(folder)
        return [f for f in files if extension is None or f.endswith(extension)]

    def save_torch(self, data, filepath):
        torch.save(data, filepath)

    def load_torch(self, filepath, map_location="cpu"):
        return torch.load(filepath, map_location=map_location)

    def save_pickle(self, data, filepath):
        joblib.dump(data, filepath)

    def load_pickle(self, filepath):
        return joblib.load(filepath)
