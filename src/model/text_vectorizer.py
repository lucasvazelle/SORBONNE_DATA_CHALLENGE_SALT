import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class TextVectorizer:
    """
    Encapsule la vectorisation du texte par TF-IDF et sa réduction de dimension via SVD.
    """
    def __init__(self, ngram_range=(1,3), max_features=20000, min_df=3, max_df=0.85, n_components=600, random_state=42):
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            use_idf=True,
            norm='l2'
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state, algorithm='arpack')

    def fit_transform(self, texts):
        """
        Apprend le modèle TF-IDF puis le modèle SVD sur les textes et retourne la matrice réduite.
        """
        X_tfidf = self.tfidf.fit_transform(texts)
        X_svd = self.svd.fit_transform(X_tfidf)
        return X_svd

    def transform(self, texts):
        """
        Applique les transformations TF-IDF et SVD sur de nouveaux textes.
        """
        X_tfidf = self.tfidf.transform(texts)
        X_svd = self.svd.transform(X_tfidf)
        return X_svd

    def save(self, tfidf_path, svd_path):
        joblib.dump(self.tfidf, tfidf_path)
        joblib.dump(self.svd, svd_path)

    def load(self, tfidf_path, svd_path):
        self.tfidf = joblib.load(tfidf_path)
        self.svd = joblib.load(svd_path)
