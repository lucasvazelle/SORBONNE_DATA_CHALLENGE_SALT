import re
from collections import Counter

class CodeBERTUtils:
    @staticmethod
    def is_valid_dot_file(filepath):
        """
        Vérifie si le fichier est un fichier DOT valide (contient "digraph" ou "graph").
        """
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip().lower()
                return bool(first_line) and ("digraph" in first_line or "graph" in first_line)
        except Exception:
            return False

    @staticmethod
    def extract_labels_from_text(dot_text):
        """
        Extrait les labels depuis un texte DOT.
        """
        return re.findall(r'label\s*=\s*"(.*?)"', dot_text)

    @staticmethod
    def extract_distinct_payloads_version_a(node_labels):
        """
        Nettoie et dédoublonne les payloads extraits des labels, triés par fréquence.
        """
        freq = Counter()
        cleaned_payloads = []
        for label in node_labels:
            parts = label.split(":")
            payload = ":".join(parts[1:]).strip() if len(parts) > 1 else label.strip()
            payload = re.sub(r"[:,\"\[\],]", " ", payload)
            payload = re.sub(r"[^a-zA-Z0-9_\-+*/\\= ]+", " ", payload)
            payload = re.sub(r"\b0x[0-9a-fA-F]+\b", '', payload)
            payload = re.sub(r'\b[a-zA-Z]+\s*=\s*0x[0-9a-fA-F]+\b', '', payload)
            payload = re.sub(r"=", "", payload)
            payload = re.sub(r"[+\-]", "", payload)
            payload = re.sub(r"\s+", " ", payload).strip()
            if payload:
                freq[payload] += 1
                cleaned_payloads.append(payload)
        distinct_ordered = []
        seen = set()
        for p in sorted(cleaned_payloads, key=lambda x: freq[x]):
            if p not in seen:
                seen.add(p)
                distinct_ordered.append(p)
        return distinct_ordered
