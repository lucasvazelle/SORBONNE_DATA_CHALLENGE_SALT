import os
import re
import numpy as np
from collections import Counter, defaultdict


class GraphFeatureExtractor:
    """
    Extrait les caractéristiques d’un graphe à partir d’un fichier JSON.
    Les features extraites incluent le texte des instructions, des statistiques sur les instructions
    et des mesures de la structure du graphe.
    """

    def __init__(self):
        pass

    def extract_features(self, filepath):
        """
        Lit le fichier, extrait les instructions et calcule plusieurs statistiques :
         - Texte concaténé
         - Nombre d’instructions, entropie, etc.
         - Informations sur le graphe (nombre de nœuds, d’arêtes, degré, profondeur, etc.)
        """
        with open(filepath, "r") as f:
            content = f.read()

        # Extraction des instructions
        inst_matches = re.findall(r"INST\s+:\s+([^:\n]+)", content)
        all_text = " ".join(inst_matches)
        inst_lower = [w.lower() for w in inst_matches]
        counter = Counter(inst_lower)

        # Quelques statistiques de base
        common_instr = ["call", "jmp", "ret", "mov", "push", "pop", "lea"]
        stats = {f"instr_{k}": counter.get(k, 0) for k in common_instr}
        stats["nb_tokens"] = len(inst_lower)
        stats["unique_instructions"] = len(set(inst_lower))
        if inst_lower:
            probs = np.array(list(counter.values())) / (len(inst_lower) + 1e-9)
            stats["instruction_entropy"] = -np.sum(probs * np.log2(probs + 1e-9))
        else:
            stats["instruction_entropy"] = 0

        # Recherche de patterns spécifiques
        stats["has_xor_zeroing"] = int(
            bool(
                re.search(
                    r"xor\s+[re]?[abcd]x,\s+[re]?[abcd]x", all_text, re.IGNORECASE
                )
            )
        )
        stats["has_getproc"] = int(
            bool(re.search(r"getprocaddress", all_text, re.IGNORECASE))
        )

        # Extraction de la structure du graphe
        edges = re.findall(r'"([0-9a-fx]+)"\s*->\s*"([0-9a-fx]+)"', content)
        nodes = set()
        out_degree = defaultdict(int)
        graph = defaultdict(list)
        for src, dst in edges:
            nodes.add(src)
            nodes.add(dst)
            out_degree[src] += 1
            graph[src].append(dst)
        stats["nb_nodes"] = len(nodes)
        stats["nb_edges"] = len(edges)
        stats["max_out_degree"] = max(out_degree.values()) if out_degree else 0

        def dfs_iterative(start_node):
            visited = set()
            stack = [(start_node, 0)]
            max_depth = 0
            while stack:
                node, depth = stack.pop()
                if node not in visited:
                    visited.add(node)
                    max_depth = max(max_depth, depth)
                    for child in graph.get(node, []):
                        if child not in visited:
                            stack.append((child, depth + 1))
            return max_depth

        entry_node = list(nodes)[0] if nodes else None
        stats["depth_max"] = dfs_iterative(entry_node) if entry_node else 0

        return all_text, stats

    def process_graphs(self, folder, selected_ids):
        """
        Parcourt le dossier `folder` et traite les fichiers dont l’identifiant (nom sans .json)
        figure dans `selected_ids`.
        Retourne deux dictionnaires : l’un pour le texte extrait et l’autre pour les statistiques.
        """
        texts = {}
        stats = {}
        for fname in os.listdir(folder):
            if fname.endswith(".json"):
                file_id = fname.replace(".json", "")
                if file_id in selected_ids:
                    path = os.path.join(folder, fname)
                    text, feature_stats = self.extract_features(path)
                    texts[file_id] = text
                    stats[file_id] = feature_stats
        return texts, stats
