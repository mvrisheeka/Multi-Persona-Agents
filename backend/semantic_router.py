import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticRouter:
    def __init__(self, dataset_path: str, threshold: float = 0.55):
        self.threshold = threshold

        print("Loading semantic model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        print("Reading dataset...")
        df = pd.read_excel(dataset_path)
        df.columns = [c.strip().lower() for c in df.columns]

        self.queries = df["query"].astype(str).tolist()
        self.personas = df["personas"].astype(str).str.lower().tolist()

        print("Embedding dataset (one-time)...")
        self.embeddings = self.model.encode(
            self.queries,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        print("Semantic router ready [DONE]")

    # -------------------------
    # predict persona
    # -------------------------
    def predict(self, query: str):

        q_emb = self.model.encode([query], normalize_embeddings=True)

        sims = cosine_similarity(q_emb, self.embeddings)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score < self.threshold:
            return None

        # dataset may contain "teacher, senior"
        persona = self.personas[best_idx].split(",")[0].strip()

        return persona, best_score
