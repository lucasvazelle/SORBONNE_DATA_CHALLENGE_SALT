import torch
from transformers import AutoTokenizer, AutoModel


class CodeBERTEmbedder:
    """
    Encapsule le modèle CodeBERT pour calculer des embeddings moyens à partir d'un batch de textes.
    """

    def __init__(self, model_name="microsoft/codebert-base", torch_dtype=torch.float16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype).to(
            self.device
        )
        self.model.eval()

    def embed_batch(self, list_texts, max_length=512):
        """
        Calcule l'embedding moyen pour chaque texte de `list_texts`.
        """
        encodings = self.tokenizer(
            list_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            outputs = self.model(**encodings)
            hidden_states = outputs.last_hidden_state
            # Moyenne pondérée par le masque d'attention
            attention_mask = (
                encodings["attention_mask"].unsqueeze(-1).expand(hidden_states.size())
            )
            summed = (hidden_states * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_pooled = summed / counts
        return mean_pooled.cpu()
