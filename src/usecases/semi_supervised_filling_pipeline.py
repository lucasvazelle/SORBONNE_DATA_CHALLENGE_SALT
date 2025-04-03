import os
import pandas as pd
from model.behavior_filler_with_kmeans import BehaviorFiller
from infrastructure.file_io_manager import FileIOManager


class SemiSupervisedFillingPipeline:
    """
    Pipeline pour remplir les valeurs manquantes des comportements logiciels
    à l'aide d'un modèle de clustering (KMeans) et des moyennes des clusters.
    Ici, on ne remplit que les colonnes du TOP 10% ayant le moins de 1 en proportion.
    """

    def __init__(
        self, training_metadata_path, prediction_metadata_path, output_dir, n_clusters=3
    ):
        self.training_metadata_path = training_metadata_path
        self.prediction_metadata_path = prediction_metadata_path
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        self.io_manager = FileIOManager()
        self.filler = BehaviorFiller(n_clusters=n_clusters)

    def run(self):
        # Charger les données d'entraînement et de prédiction
        metadata_train = self.io_manager.read_csv(self.training_metadata_path, sep=";")
        metadata_predict = pd.read_excel(self.prediction_metadata_path)

        # Calculer la proportion de 1 pour chaque comportement (colonnes à partir de la 2ème)
        counts = metadata_train.iloc[:, 1:].sum(axis=0)
        total_rows = metadata_train.shape[0]
        percentages = counts / total_rows
        # Sélectionner le TOP 10% des colonnes ayant le moins de 1
        num_columns = len(percentages)
        num_top = max(1, int(0.1 * num_columns))
        known_behaviors = percentages.nsmallest(num_top).index.tolist()
        print("Known behaviors (TOP 10% avec le moins de 1) :", known_behaviors)

        # Entraîner le modèle KMeans sur les données d'entraînement
        self.filler.train_kmeans(metadata_train)
        print("Moyennes des clusters:")
        print(self.filler.cluster_means)

        # Remplir uniquement les colonnes identifiées (known_behaviors)
        results_df = self.filler.fill_dataset(
            metadata_predict, columns_to_fill=known_behaviors
        )
        output_path = os.path.join(
            self.output_dir, f"metadata_predict_filled_{self.n_clusters}.xlsx"
        )
        results_df.to_excel(output_path, index=False)
        print(f"Résultats sauvegardés dans {output_path}")
