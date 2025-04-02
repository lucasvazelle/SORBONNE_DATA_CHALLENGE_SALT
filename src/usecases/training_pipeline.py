import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from infrastructure.file_io_manager import FileIOManager
from domain.graph_feature_extractor import GraphFeatureExtractor
from model.text_vectorizer import TextVectorizer
from model.xgb_classifier import XGBMultiLabelClassifier

class TrainingPipeline:
    """
    Pipeline de formation complet :
      - Extraction des features des graphes
      - Transformation du texte par TF-IDF + SVD
      - Fusion avec les métadonnées
      - Entraînement du classifieur XGBoost multilabel
    """
    def __init__(self, train_csv, train_graph_dir, embedding_dir, output_dir, sample_size=None):
        self.train_csv = train_csv
        self.train_graph_dir = train_graph_dir
        self.embedding_dir = embedding_dir
        self.output_dir = output_dir
        self.sample_size = sample_size

        self.io_manager = FileIOManager()
        self.feature_extractor = GraphFeatureExtractor()
        self.text_vectorizer = TextVectorizer()
        self.classifier = XGBMultiLabelClassifier()

    def get_embedded_ids(self):
        files = self.io_manager.list_files(self.embedding_dir, extension=".pt")
        all_ids = set()
        for fname in files:
            if fname.startswith("embedding_"):
                try:
                    data = self.io_manager.load_torch(os.path.join(self.embedding_dir, fname))
                    all_ids.update(data.get("ids", []))
                except Exception:
                    continue
        return all_ids

    def run(self):
        # Chargement des métadonnées
        train_metadata = self.io_manager.read_csv(self.train_csv, sep=';')
        train_metadata['name'] = train_metadata['name'].astype(str)
        embedded_ids = self.get_embedded_ids()
        if self.sample_size:
            sampled_metadata = train_metadata[train_metadata['name'].isin(list(embedded_ids))].sample(n=self.sample_size, random_state=42)
        else:
            sampled_metadata = train_metadata[train_metadata['name'].isin(embedded_ids)]
        selected_ids = set(sampled_metadata['name'])

        # Extraction des features de graphes
        texts, stats = self.feature_extractor.process_graphs(self.train_graph_dir, selected_ids)
        texts_df = pd.Series(texts).rename_axis('name').reset_index(name='text')
        stats_df = pd.DataFrame.from_dict(stats, orient='index').reset_index().rename(columns={'index': 'name'})
        combined_df = pd.merge(texts_df, stats_df, on='name')
        full_train_df = pd.merge(combined_df, sampled_metadata, on='name')

        # Transformation du texte par TF-IDF et SVD
        X_svd = self.text_vectorizer.fit_transform(full_train_df['text'])
        # Vous pouvez ajouter ici d'autres features si nécessaire (par ex. embeddings déjà calculés)

        # Pour cet exemple, supposons que les features finales sont X_svd et que la cible est dans la colonne 'target'
        feature_cols = []  # X_svd sera directement utilisé
        X = X_svd
        y = full_train_df[['target']].astype(int).values  # Adaptez si plusieurs cibles

        # Séparation train/validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Entraînement du classifieur
        self.classifier.train(X_train, y_train)
        f1 = self.classifier.evaluate(X_val, y_val)
        print("F1 Macro sur validation:", f1)

        # Sauvegarde du modèle et des objets de transformation
        model_path = os.path.join(self.output_dir, "xgb_model_full.pkl")
        self.classifier.save(model_path)
        tfidf_path = os.path.join(self.output_dir, "tfidf_vectorizer.pkl")
        svd_path = os.path.join(self.output_dir, "svd_model.pkl")
        self.text_vectorizer.save(tfidf_path, svd_path)
        print(f"Modèle et objets sauvegardés dans {self.output_dir}")
