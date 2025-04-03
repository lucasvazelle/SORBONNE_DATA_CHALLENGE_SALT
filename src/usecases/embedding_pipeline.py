import os
import json
from datetime import datetime
from tqdm import tqdm
import torch
from infrastructure.file_io_manager import FileIOManager
from domain.codebert_utils import CodeBERTUtils
from model.codebert_embedder import CodeBERTEmbedder


class EmbeddingPipeline:
    """
    Pipeline d'extraction d'embeddings CodeBERT pour un ensemble de graphes.
    Permet de traiter un dossier de fichiers et de sauvegarder les embeddings par batch.
    """

    def __init__(self, json_folder, save_dir, batch_size=32):
        self.json_folder = json_folder
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.io_manager = FileIOManager()
        self.embedder = CodeBERTEmbedder()
        self.utils = CodeBERTUtils()

    def get_all_embedded_ids(self):
        files = self.io_manager.list_files(self.save_dir, extension=".pt")
        all_ids = set()
        for fname in files:
            if fname.startswith("embedding_"):
                try:
                    data = self.io_manager.load_torch(
                        os.path.join(self.save_dir, fname)
                    )
                    all_ids.update(data.get("ids", []))
                except Exception:
                    continue
        return all_ids

    def encode_graphs_in_size_range(
        self, min_kb, max_kb, custom_file_list=None, custom_output_name=None
    ):
        embedded_ids = self.get_all_embedded_ids()
        all_files = []
        if custom_file_list:
            all_files = [
                (path, os.path.getsize(path) / 1024.0) for path in custom_file_list
            ]
        else:
            for fname in self.io_manager.list_files(
                self.json_folder, extension=".json"
            ):
                path = os.path.join(self.json_folder, fname)
                size_kb = os.path.getsize(path) / 1024.0
                if min_kb <= size_kb < max_kb:
                    graph_id = fname.replace(".json", "")
                    if graph_id not in embedded_ids:
                        all_files.append((path, size_kb))
        all_files.sort(key=lambda x: x[1])
        print(f"Fichiers à traiter ({min_kb}-{max_kb} KB): {len(all_files)}")

        buffer_instr, buffer_ids = [], []
        embeddings, ids, failed = [], [], []

        for path, size_kb in tqdm(all_files):
            if not self.utils.is_valid_dot_file(path):
                failed.append(os.path.basename(path))
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    dot_text = f.read()
                labels = self.utils.extract_labels_from_text(dot_text)
                distinct_payloads = self.utils.extract_distinct_payloads_version_a(
                    labels
                )
                combined_instr = " ".join(distinct_payloads)[:2560]
                file_id = os.path.basename(path).replace(".json", "")
                buffer_instr.append(combined_instr)
                buffer_ids.append(file_id)
                if len(buffer_instr) >= self.batch_size:
                    batch_emb = self.embedder.embed_batch(buffer_instr)
                    embeddings.extend(batch_emb)
                    ids.extend(buffer_ids)
                    buffer_instr.clear()
                    buffer_ids.clear()
            except Exception as e:
                print(f"Erreur avec {os.path.basename(path)}: {e}")
                failed.append(os.path.basename(path))
        if buffer_instr:
            batch_emb = self.embedder.embed_batch(buffer_instr)
            embeddings.extend(batch_emb)
            ids.extend(buffer_ids)

        if embeddings:
            tensor = torch.stack(embeddings)
            output_name = (
                custom_output_name
                if custom_output_name
                else f"embedding_{min_kb}_{max_kb}_KB.pt"
            )
            self.io_manager.save_torch(
                {"ids": ids, "embeddings": tensor},
                os.path.join(self.save_dir, output_name),
            )
            print(f"Sauvegarde de {len(ids)} graphes dans {output_name}")
        if failed:
            log_path = os.path.join(self.save_dir, f"log_{min_kb}_{max_kb}_KB.json")
            self.io_manager.save_json(
                {"failed": failed, "timestamp": datetime.now().isoformat()}, log_path
            )
            print(f"{len(failed)} erreurs, log sauvegardé dans {log_path}")
