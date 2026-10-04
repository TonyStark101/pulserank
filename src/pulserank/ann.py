import numpy as np


class HNSWIndex:
    """HNSW maximum-inner-product index for two-tower vectors plus item bias."""

    def __init__(self, retrieval, ef_construction=200, m=32, ef_search=300):
        import hnswlib

        item_vectors = np.column_stack([retrieval.item_embeddings, retrieval.item_bias]).astype(np.float32)
        self.user_embeddings = retrieval.user_embeddings
        self.index = hnswlib.Index(space="ip", dim=item_vectors.shape[1])
        self.index.init_index(
            max_elements=len(item_vectors), ef_construction=ef_construction, M=m, random_seed=17,
        )
        self.index.add_items(item_vectors, np.arange(len(item_vectors)), num_threads=1)
        self.index.set_ef(ef_search)

    def query(self, user_index, count, allowed=None):
        query = np.append(self.user_embeddings[user_index], 1.0).astype(np.float32)
        allowed = None if allowed is None else set(int(value) for value in allowed)
        available = len(allowed) if allowed is not None else self.index.get_current_count()
        count = min(count, available)
        if count <= 0:
            return np.array([], dtype=np.int64)
        filter_function = None if allowed is None else lambda label: label in allowed
        labels, _ = self.index.knn_query(
            query, k=count, num_threads=1, filter=filter_function,
        )
        return labels[0].astype(np.int64)
