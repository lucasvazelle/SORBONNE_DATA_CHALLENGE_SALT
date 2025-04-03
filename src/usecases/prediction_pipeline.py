import os
import pandas as pd
import joblib
from infrastructure.file_io_manager import FileIOManager
from model.xgb_classifier import XGBMultiLabelClassifier


class PredictionPipeline:
    """
    Pipeline de prédiction sur le test set :
      - Chargement des modèles de transformation (TF-IDF, SVD) et du classifieur
      - Transformation des textes test
      - Prédiction et génération du fichier de soumission
    """

    def __init__(
        self,
        tfidf_path,
        svd_path,
        model_path,
        test_data_path,
        submission_template,
        output_dir,
    ):
        self.tfidf_path = tfidf_path
        self.svd_path = svd_path
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.submission_template = submission_template
        self.output_dir = output_dir
        self.io_manager = FileIOManager()

    def run(self):
        # Chargement des modèles de transformation et du classifieur
        vectorizer = joblib.load(self.tfidf_path)
        svd = joblib.load(self.svd_path)
        classifier = XGBMultiLabelClassifier()
        classifier.load(self.model_path)

        # Chargement des données test (supposées être un DataFrame sauvegardé en pickle)
        test_df = self.io_manager.load_pickle(self.test_data_path)
        test_df["text_short"] = test_df["text"].str.slice(0, 200000)
        X_tfidf = vectorizer.transform(test_df["text_short"])
        X_svd = svd.transform(X_tfidf)
        X_test = X_svd  # Fusionnez ici avec d'autres features si nécessaire

        # Prédiction
        y_pred_proba = classifier.predict_proba(X_test)
        if classifier.thresholds is not None:
            y_pred = (y_pred_proba > classifier.thresholds).astype(int)
        else:
            y_pred = (y_pred_proba > 0.5).astype(int)

        # Création du fichier de soumission
        submission_df = pd.read_excel(self.submission_template)
        test_ids_pred = test_df["name"].tolist()
        submission_array = pd.DataFrame(y_pred, columns=submission_df.columns[1:])
        submission_array.insert(0, "name", test_ids_pred)
        missing_ids = set(submission_df["name"].astype(str)) - set(test_ids_pred)
        if missing_ids:
            zero_df = pd.DataFrame(
                0, index=range(len(missing_ids)), columns=submission_df.columns
            )
            zero_df["name"] = list(missing_ids)
            submission_array = pd.concat([submission_array, zero_df], axis=0)
        submission_array = (
            submission_array.set_index("name")
            .reindex(submission_df["name"])
            .reset_index()
        )
        submission_file = os.path.join(self.output_dir, "submission.xlsx")
        submission_array.to_excel(submission_file, index=False)
        print(f"Fichier de soumission créé : {submission_file}")
