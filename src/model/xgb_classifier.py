import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
from sklearn import set_config


class XGBMultiLabelClassifier:
    """
    Encapsule un classifieur XGBoost pour la classification multilabel,
    avec optimisation bayésienne des seuils.
    """

    def __init__(self, n_jobs=-1, random_state=42, **xgb_params):
        self.xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="gpu_hist",
            reg_alpha=xgb_params.get("reg_alpha", 0.5),
            reg_lambda=xgb_params.get("reg_lambda", 1.0),
            gamma=xgb_params.get("gamma", 0.2),
            use_label_encoder=False,
            verbosity=0,
        )
        self.model = OneVsRestClassifier(self.xgb, n_jobs=n_jobs)
        set_config(enable_metadata_routing=True)
        self.thresholds = None

    def train(self, X_train, y_train, sample_weight=None):
        self.model.fit(X_train, y_train, sample_weight=sample_weight)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        if self.thresholds is None:
            return (proba > 0.5).astype(int)
        return (proba > self.thresholds).astype(int)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

    def optimize_thresholds(self, y_true, y_proba, init_points=5, n_iter=10):
        thresholds = []
        for i in range(y_true.shape[1]):
            y_true_i = y_true[:, i]
            y_proba_i = y_proba[:, i]

            def f(t):
                y_pred_bin = (y_proba_i > t).astype(int)
                return f1_score(y_true_i, y_pred_bin, zero_division=1)

            optimizer = BayesianOptimization(
                f=f, pbounds={"t": (0.05, 0.95)}, random_state=42, verbose=0
            )
            optimizer.maximize(init_points=init_points, n_iter=n_iter)
            best_t = optimizer.max["params"]["t"]
            thresholds.append(best_t)
        self.thresholds = np.array(thresholds)
        return self.thresholds

    def save(self, filepath):
        joblib.dump({"model": self.model, "thresholds": self.thresholds}, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.model = data.get("model")
        self.thresholds = data.get("thresholds")
