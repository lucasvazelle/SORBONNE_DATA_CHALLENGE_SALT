# cli/main.py
import argparse
from usecases.embedding_pipeline import EmbeddingPipeline
from usecases.training_pipeline import TrainingPipeline
from usecases.prediction_pipeline import PredictionPipeline
from usecases.semi_supervised_filling_pipeline import SemiSupervisedFillingPipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline complet pour extraction d'embeddings, formation, prédiction et remplissage semi-supervisé.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed_parser = subparsers.add_parser("embed", help="Générer des embeddings CodeBERT.")
    embed_parser.add_argument("--json_folder", required=True, help="Répertoire des fichiers JSON.")
    embed_parser.add_argument("--save_dir", required=True, help="Répertoire de sauvegarde des embeddings.")
    embed_parser.add_argument("--min_kb", type=float, required=True, help="Taille minimale en KB.")
    embed_parser.add_argument("--max_kb", type=float, required=True, help="Taille maximale en KB.")
    embed_parser.add_argument("--batch_size", type=int, default=32, help="Taille du batch.")

    train_parser = subparsers.add_parser("train", help="Exécuter le pipeline d'entraînement.")
    train_parser.add_argument("--train_csv", required=True, help="Chemin vers le CSV d'entraînement.")
    train_parser.add_argument("--train_graph_dir", required=True, help="Répertoire des graphes d'entraînement.")
    train_parser.add_argument("--embedding_dir", required=True, help="Répertoire contenant les embeddings existants.")
    train_parser.add_argument("--output_dir", required=True, help="Répertoire de sortie pour sauvegarder les modèles.")
    train_parser.add_argument("--sample_size", type=int, default=None, help="Taille de l'échantillon (optionnel).")

    pred_parser = subparsers.add_parser("predict", help="Exécuter le pipeline de prédiction.")
    pred_parser.add_argument("--tfidf_path", required=True, help="Chemin vers le TF-IDF sauvegardé.")
    pred_parser.add_argument("--svd_path", required=True, help="Chemin vers le modèle SVD sauvegardé.")
    pred_parser.add_argument("--model_path", required=True, help="Chemin vers le modèle XGBoost sauvegardé.")
    pred_parser.add_argument("--test_data_path", required=True, help="Chemin vers les données test (pickle).")
    pred_parser.add_argument("--submission_template", required=True, help="Chemin vers le template Excel de soumission.")
    pred_parser.add_argument("--output_dir", required=True, help="Répertoire de sortie pour la soumission.")

    fill_parser = subparsers.add_parser("fill", help="Exécuter le pipeline de remplissage semi-supervisé.")
    fill_parser.add_argument("--training_metadata", required=True, help="Chemin vers le CSV d'entraînement (metadata).")
    fill_parser.add_argument("--prediction_metadata", required=True, help="Chemin vers le fichier Excel de prédiction.")
    fill_parser.add_argument("--output_dir", required=True, help="Répertoire de sortie pour le résultat.")
    fill_parser.add_argument("--n_clusters", type=int, default=3, help="Nombre de clusters à utiliser.")

    args = parser.parse_args()

    if args.command == "embed":
        pipeline = EmbeddingPipeline(json_folder=args.json_folder, save_dir=args.save_dir, batch_size=args.batch_size)
        pipeline.encode_graphs_in_size_range(min_kb=args.min_kb, max_kb=args.max_kb)
    elif args.command == "train":
        pipeline = TrainingPipeline(
            train_csv=args.train_csv,
            train_graph_dir=args.train_graph_dir,
            embedding_dir=args.embedding_dir,
            output_dir=args.output_dir,
            sample_size=args.sample_size
        )
        pipeline.run()
    elif args.command == "predict":
        pipeline = PredictionPipeline(
            tfidf_path=args.tfidf_path,
            svd_path=args.svd_path,
            model_path=args.model_path,
            test_data_path=args.test_data_path,
            submission_template=args.submission_template,
            output_dir=args.output_dir
        )
        pipeline.run()
    elif args.command == "fill":
        pipeline = SemiSupervisedFillingPipeline(
            training_metadata_path=args.training_metadata,
            prediction_metadata_path=args.prediction_metadata,
            output_dir=args.output_dir,
            n_clusters=args.n_clusters
        )
        pipeline.run()

if __name__ == "__main__":
    main()
