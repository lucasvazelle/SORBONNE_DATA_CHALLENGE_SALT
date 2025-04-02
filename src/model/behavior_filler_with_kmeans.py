import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class BehaviorFiller:
    """
    Algorithme pour remplir les valeurs manquantes (binary) pour de nouveaux logiciels.
    Utilise KMeans pour identifier le cluster le plus proche et complète les comportements manquants
    en appliquant la valeur majoritaire du cluster.
    """
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_means = None

    def train_kmeans(self, metadata_train):
        """
        Entraîne un modèle KMeans sur les données binaires (en excluant la première colonne d'ID).
        Retourne le modèle et les moyennes des clusters.
        """
        X = metadata_train.iloc[:, 1:]  # suppose que la première colonne est l'ID
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = self.kmeans.fit_predict(X)
        metadata_train = metadata_train.copy()
        metadata_train["Cluster"] = clusters
        self.cluster_means = metadata_train.iloc[:, 1:].groupby("Cluster").mean()
        return self.kmeans, self.cluster_means

    def fill_missing_behaviors(self, known_behaviors):
        """
        Remplit les valeurs manquantes (indiquées par -1) dans known_behaviors.
        Retourne un tuple (filled_behaviors, best_cluster) où best_cluster est le cluster le plus proche.
        """
        known_behaviors = np.array(known_behaviors).reshape(1, -1)
        mask_known = (known_behaviors != -1).reshape(-1)
        distances = []
        for i, cluster_center in enumerate(self.kmeans.cluster_centers_):
            cluster_center_filtered = cluster_center[mask_known]
            known_filtered = known_behaviors[0][mask_known]
            distance = np.linalg.norm(cluster_center_filtered - known_filtered)
            distances.append(distance)
        best_cluster = np.argmin(distances)
        filled_behaviors = known_behaviors.copy()
        for i in range(len(filled_behaviors[0])):
            if filled_behaviors[0][i] == -1:
                filled_behaviors[0][i] = 1 if self.cluster_means.iloc[best_cluster, i] > 0.5 else 0
        return filled_behaviors[0], best_cluster

    def fill_dataset(self, metadata_predict):
        """
        Pour chaque ligne de metadata_predict (contenant l'ID et les comportements),
        complète les valeurs manquantes et ajoute une colonne 'predicted_cluster'.
        Retourne un DataFrame avec les comportements complétés.
        """
        results = []
        for idx, row in metadata_predict.iterrows():
            known_behaviors = row.iloc[1:].values  # la première colonne est l'ID
            filled, cluster_idx = self.fill_missing_behaviors(known_behaviors)
            result_row = [row.iloc[0]] + list(filled) + [cluster_idx]
            results.append(result_row)
        columns = [metadata_predict.columns[0]] + list(metadata_predict.columns[1:]) + ['predicted_cluster']
        return pd.DataFrame(results, columns=columns)
