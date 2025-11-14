"""
RAG: Retrieval-Augmented Generation
Enhance LLM with external knowledge retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DenseRetriever(nn.Module):
    """
    Dense retrieval using dual encoders

    Architecture:
        Query → Query Encoder → Query Embedding
        Documents → Document Encoder → Document Embeddings
        Similarity: cosine(query_emb, doc_embs)
    """

    def __init__(self, vocab_size, embed_dim=768, hidden_dim=768):
        super().__init__()

        # Query encoder (Transformer-based)
        self.query_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=6
        )

        # Document encoder (shared or separate)
        self.doc_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=6
        )

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Projection to final embedding
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.doc_projection = nn.Linear(embed_dim, embed_dim)

    def encode_query(self, query_tokens):
        """
        Args:
            query_tokens: (B, L_query)

        Returns:
            query_emb: (B, embed_dim)
        """
        emb = self.embedding(query_tokens)
        encoded = self.query_encoder(emb)
        pooled = encoded.mean(dim=1)  # Mean pooling
        query_emb = self.query_projection(pooled)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        return query_emb

    def encode_docs(self, doc_tokens):
        """
        Args:
            doc_tokens: (B, L_doc)

        Returns:
            doc_emb: (B, embed_dim)
        """
        emb = self.embedding(doc_tokens)
        encoded = self.doc_encoder(emb)
        pooled = encoded.mean(dim=1)
        doc_emb = self.doc_projection(pooled)
        doc_emb = F.normalize(doc_emb, p=2, dim=-1)
        return doc_emb

    def retrieve(self, query_emb, doc_embs, top_k=5):
        """
        Retrieve top-k documents

        Args:
            query_emb: (B_query, embed_dim)
            doc_embs: (N_docs, embed_dim)
            top_k: Number of documents to retrieve

        Returns:
            top_indices: (B_query, top_k)
            top_scores: (B_query, top_k)
        """
        # Compute similarity
        scores = torch.matmul(query_emb, doc_embs.T)  # (B_query, N_docs)

        # Top-k
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=-1)

        return top_indices, top_scores


class SparseRetriever:
    """
    BM25-based sparse retrieval (traditional IR)

    Simpler alternative to dense retrieval
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = None
        self.avg_doc_len = None
        self.doc_lens = None
        self.idf = None

    def fit(self, documents):
        """
        Compute BM25 statistics

        Args:
            documents: List of tokenized documents
        """
        N = len(documents)
        self.doc_lens = [len(doc) for doc in documents]
        self.avg_doc_len = sum(self.doc_lens) / N

        # Compute document frequencies
        self.doc_freqs = {}
        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Compute IDF
        self.idf = {}
        for term, df in self.doc_freqs.items():
            self.idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query, doc, doc_len):
        """Compute BM25 score for a query-document pair"""
        score = 0.0
        for term in query:
            if term not in self.idf:
                continue

            # Term frequency in document
            tf = doc.count(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += self.idf[term] * numerator / denominator

        return score


class RAG:
    """
    Retrieval-Augmented Generation

    Pipeline:
        1. Query → Retriever → Top-K Documents
        2. Concatenate: [Query, Retrieved Docs]
        3. Generator (LLM) → Answer
    """

    def __init__(self, retriever, generator, doc_embeddings, documents, top_k=5):
        """
        Args:
            retriever: Dense retriever model
            generator: LLM for generation (e.g., GPT-2, T5)
            doc_embeddings: (N_docs, embed_dim) - precomputed doc embeddings
            documents: List of document texts
            top_k: Number of docs to retrieve
        """
        self.retriever = retriever
        self.generator = generator
        self.doc_embeddings = doc_embeddings
        self.documents = documents
        self.top_k = top_k

    def retrieve(self, query_tokens):
        """
        Retrieve relevant documents

        Args:
            query_tokens: (B, L_query)

        Returns:
            retrieved_docs: List of lists of document texts
            scores: (B, top_k) - relevance scores
        """
        with torch.no_grad():
            # Encode query
            query_emb = self.retriever.encode_query(query_tokens)

            # Retrieve top-k documents
            top_indices, top_scores = self.retriever.retrieve(
                query_emb, self.doc_embeddings, self.top_k
            )

        # Get document texts
        batch_size = query_tokens.size(0)
        retrieved_docs = []

        for i in range(batch_size):
            docs = [self.documents[idx] for idx in top_indices[i]]
            retrieved_docs.append(docs)

        return retrieved_docs, top_scores

    def generate(self, query, retrieved_docs, max_length=100):
        """
        Generate answer using query + retrieved documents

        Args:
            query: Query text
            retrieved_docs: List of retrieved document texts
            max_length: Max generation length

        Returns:
            answer: Generated answer
        """
        # Construct prompt with retrieved context
        context = "\n\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate using LLM (pseudo-code, depends on LLM API)
        # answer = self.generator.generate(prompt, max_length=max_length)

        return prompt  # In practice, return generated answer

    def __call__(self, query_tokens, query_text):
        """
        Full RAG pipeline

        Args:
            query_tokens: (B, L_query) - tokenized query
            query_text: Original query text

        Returns:
            answers: Generated answers
            retrieved_docs: Retrieved documents
        """
        # Retrieve
        retrieved_docs, scores = self.retrieve(query_tokens)

        # Generate
        answers = []
        for i, docs in enumerate(retrieved_docs):
            answer = self.generate(query_text[i], docs)
            answers.append(answer)

        return answers, retrieved_docs


def build_doc_index(retriever, documents, tokenizer, batch_size=32):
    """
    Precompute document embeddings for efficient retrieval

    Args:
        retriever: Dense retriever model
        documents: List of document texts
        tokenizer: Tokenizer for documents
        batch_size: Batch size for encoding

    Returns:
        doc_embeddings: (N_docs, embed_dim)
    """
    retriever.eval()
    doc_embeddings = []

    with torch.no_grad():
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            # Tokenize
            doc_tokens = tokenizer(batch_docs, padding=True, truncation=True, return_tensors='pt')

            # Encode
            batch_embs = retriever.encode_docs(doc_tokens['input_ids'])
            doc_embeddings.append(batch_embs)

    doc_embeddings = torch.cat(doc_embeddings, dim=0)
    return doc_embeddings


if __name__ == "__main__":
    print("=" * 60)
    print("RAG: Retrieval-Augmented Generation")
    print("=" * 60)

    # Create retriever
    vocab_size = 30000
    embed_dim = 768

    retriever = DenseRetriever(vocab_size=vocab_size, embed_dim=embed_dim)

    print(f"\nDense Retriever:")
    total_params = sum(p.numel() for p in retriever.parameters())
    print(f"Parameters: {total_params:,}")

    # Example: Encode query and documents
    batch_size = 2
    num_docs = 100

    query_tokens = torch.randint(0, vocab_size, (batch_size, 20))
    doc_tokens = torch.randint(0, vocab_size, (num_docs, 50))

    print(f"\nExample:")
    print(f"Query tokens: {query_tokens.shape}")
    print(f"Document tokens: {doc_tokens.shape}")

    # Encode
    query_emb = retriever.encode_query(query_tokens)
    doc_embs = retriever.encode_docs(doc_tokens)

    print(f"Query embeddings: {query_emb.shape}")
    print(f"Document embeddings: {doc_embs.shape}")

    # Retrieve
    top_k = 5
    top_indices, top_scores = retriever.retrieve(query_emb, doc_embs, top_k=top_k)

    print(f"\nRetrieval:")
    print(f"Top-{top_k} indices: {top_indices.shape}")
    print(f"Top-{top_k} scores: {top_scores.shape}")
    print(f"Example scores: {top_scores[0].tolist()}")

    # Example: BM25 sparse retrieval
    print("\n" + "=" * 60)
    print("BM25 Sparse Retrieval")
    print("=" * 60)

    documents = [
        ["cat", "dog", "animal"],
        ["machine", "learning", "ai"],
        ["cat", "meow", "pet"],
    ]

    bm25 = SparseRetriever()
    bm25.fit(documents)

    query = ["cat", "pet"]
    scores = [bm25.score(query, doc, len(doc)) for doc in documents]

    print(f"Query: {query}")
    print(f"BM25 scores: {scores}")
    print(f"Top document: {documents[np.argmax(scores)]}")

    print("\n" + "=" * 60)
    print("RAG Complete!")
    print("=" * 60)
    print("\nKey Concepts:")
    print("✓ Dense Retrieval: Encode query & docs → similarity search")
    print("✓ Sparse Retrieval: BM25, TF-IDF (traditional IR)")
    print("✓ Hybrid: Combine dense + sparse for best results")
    print("✓ Generation: LLM conditioned on retrieved context")
    print("✓ Benefits: Factuality, up-to-date info, citations")

    print("\n" + "=" * 60)
    print("RAG Pipeline:")
    print("=" * 60)
    print("""
    1. Index documents (precompute embeddings)
       doc_embeddings = retriever.encode_docs(all_documents)

    2. At query time:
       a. Encode query
       b. Retrieve top-k documents
       c. Construct prompt: [context + query]
       d. Generate answer with LLM

    3. Example:
       query = "What is RAG?"
       docs, scores = rag.retrieve(query)
       answer = llm.generate(f"Context: {docs}\\n\\nQ: {query}\\nA:")
    """)
