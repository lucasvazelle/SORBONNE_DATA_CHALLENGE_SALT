# SORBONNE_DATA_CHALLENGE_SALTmon_projet/

Vous y retrouverez :

- L’extraction d’embeddings avec CodeBERT 

- L’extraction des features des graphes 

- La vectorisation du texte via TF‑IDF et la réduction de dimension par SVD 

- La concaténation finale, l’entraînement d’un modèle XGBoost multilabel avec optimisation des seuils 

- La prédiction sur le test et la génération de la soumission 


cli (point d'entrée) : en vert

domain (logique métier, extraction CodeBERT et features) : en bleu

infrastructure (gestion des I/O) : en violet

model (modélisation : CodeBERT embedder, TF‑IDF/SVD, classifieur) : en rouge

usecases (orchestration du pipeline : entraînement et prédiction) : en orange

tests (tests unitaires) : en gris

requirements.txt et setup.py (fichiers de configuration) : en or
mon_projet/
├── <span style="color:green;">cli/</span>
│   ├── <span style="color:green;">__init__.py</span>
│   └── <span style="color:green;">main.py</span>
├── <span style="color:blue;">domain/</span>
│   ├── <span style="color:blue;">__init__.py</span>
│   ├── <span style="color:blue;">codebert_utils.py</span>
│   └── <span style="color:blue;">graph_feature_extractor.py</span>
├── <span style="color:purple;">infrastructure/</span>
│   ├── <span style="color:purple;">__init__.py</span>
│   └── <span style="color:purple;">file_io_manager.py</span>
├── <span style="color:red;">model/</span>
│   ├── <span style="color:red;">__init__.py</span>
│   ├── <span style="color:red;">codebert_embedder.py</span>
│   └── <span style="color:red;">xgb_classifier.py</span>
├── <span style="color:orange;">usecases/</span>
│   ├── <span style="color:orange;">__init__.py</span>
│   ├── <span style="color:orange;">training_pipeline.py</span>
│   └── <span style="color:orange;">prediction_pipeline.py</span>
├── <span style="color:gray;">tests/</span>
│   ├── <span style="color:gray;">__init__.py</span>
│   ├── <span style="color:gray;">test_codebert_embedder.py</span>
│   ├── <span style="color:gray;">test_xgb_classifier.py</span>
│   ├── <span style="color:gray;">test_graph_feature_extractor.py</span>
│   ├── <span style="color:gray;">test_file_io_manager.py</span>
│   └── <span style="color:gray;">test_training_pipeline.py</span>
├── <span style="color:goldenrod;">requirements.txt</span>
└── <span style="color:goldenrod;">setup.py</span>
