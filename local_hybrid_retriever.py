"""
Local Hybrid Retriever - Uses your own Pinecone account + local Elasticsearch
No AWS dependencies required.
"""

import os
import torch
from pinecone import Pinecone, ServerlessSpec
from elasticsearch import Elasticsearch
from transformers import AutoModel, AutoTokenizer


class LocalHybridRetriever:
    def __init__(
        self,
        pinecone_api_key=None,
        pinecone_index_name="ragenta-index",
        elasticsearch_url="http://localhost:9200",
        elasticsearch_index_name="ragenta-docs",
        alpha=0.65,
        top_k=20,
        embedding_model="intfloat/e5-base-v2"
    ):
        """
        Initialize a hybrid retriever using your own Pinecone + local Elasticsearch.

        Args:
            pinecone_api_key: Your Pinecone API key (or set PINECONE_API_KEY env var)
            pinecone_index_name: Name of your Pinecone index
            elasticsearch_url: URL of your local Elasticsearch instance
            elasticsearch_index_name: Name of your Elasticsearch index
            alpha: Weight for semantic search (0-1). Higher = more semantic weight.
            top_k: Number of documents to retrieve
            embedding_model: Sentence transformer model for embeddings
        """
        self.alpha = alpha
        self.top_k = top_k
        self.pinecone_index_name = pinecone_index_name
        self.elasticsearch_index_name = elasticsearch_index_name

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.model_name = embedding_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Embedding model loaded on {self.device}")

        # Initialize Pinecone
        print("Connecting to Pinecone...")
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError(
                "Pinecone API key required. Set PINECONE_API_KEY environment variable "
                "or pass pinecone_api_key parameter."
            )

        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Check if index exists, create if not
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if pinecone_index_name not in existing_indexes:
            print(f"Creating Pinecone index: {pinecone_index_name}")
            self.pc.create_index(
                name=pinecone_index_name,
                dimension=768,  # e5-base-v2 dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        self.pinecone_index = self.pc.Index(name=pinecone_index_name)
        print(f"Connected to Pinecone index: {pinecone_index_name}")

        # Initialize Elasticsearch
        print(f"Connecting to Elasticsearch at {elasticsearch_url}...")
        self.es_client = Elasticsearch(
            elasticsearch_url,
            verify_certs=False,
            ssl_show_warn=False,
            request_timeout=30
        )

        # Check Elasticsearch connection
        try:
            if not self.es_client.ping():
                raise ConnectionError(
                    f"Cannot connect to Elasticsearch at {elasticsearch_url}. "
                    "Make sure Elasticsearch is running."
                )
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Elasticsearch at {elasticsearch_url}. "
                f"Error: {e}"
            )

        # Create index if not exists
        if not self.es_client.indices.exists(index=elasticsearch_index_name):
            print(f"Creating Elasticsearch index: {elasticsearch_index_name}")
            self.es_client.indices.create(
                index=elasticsearch_index_name,
                body={
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "doc_id": {"type": "keyword"}
                        }
                    }
                }
            )

        print("Local hybrid retriever initialized successfully!")

    def _embed_query(self, query):
        """Create embeddings for a query."""
        query_with_prefix = f"query: {query}"

        with torch.no_grad():
            inputs = self.tokenizer(
                [query_with_prefix], return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            # Use average pooling
            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().tolist()

    def _embed_document(self, text):
        """Create embeddings for a document."""
        doc_with_prefix = f"passage: {text}"

        with torch.no_grad():
            inputs = self.tokenizer(
                [doc_with_prefix], return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)

            attention_mask = inputs["attention_mask"]
            last_hidden = outputs.last_hidden_state.masked_fill(
                ~attention_mask[..., None].bool(), 0.0
            )
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().tolist()

    def index_documents(self, documents, batch_size=100):
        """
        Index documents into both Pinecone and Elasticsearch.

        Args:
            documents: List of dicts with 'id' and 'text' keys
            batch_size: Batch size for indexing
        """
        print(f"Indexing {len(documents)} documents...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Prepare Pinecone vectors
            pinecone_vectors = []
            es_bulk_body = []

            for doc in batch:
                doc_id = str(doc["id"])
                text = doc["text"]

                # Create embedding
                embedding = self._embed_document(text)

                # Pinecone vector
                pinecone_vectors.append({
                    "id": doc_id,
                    "values": embedding,
                    "metadata": {"text": text[:1000]}  # Store first 1000 chars
                })

                # Elasticsearch document
                es_bulk_body.append({"index": {"_index": self.elasticsearch_index_name, "_id": doc_id}})
                es_bulk_body.append({"text": text, "doc_id": doc_id})

            # Upsert to Pinecone
            self.pinecone_index.upsert(vectors=pinecone_vectors)

            # Bulk index to Elasticsearch
            if es_bulk_body:
                self.es_client.bulk(body=es_bulk_body, refresh=True)

            print(f"Indexed batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}")

        print("Indexing complete!")

    def _normalize_scores(self, results, score_key):
        """Normalize scores to range 0-1."""
        if not results:
            return []

        scores = [res[score_key] for res in results]
        min_score, max_score = min(scores), max(scores)

        for res in results:
            if max_score > min_score:
                res["normalized_score"] = (res[score_key] - min_score) / (max_score - min_score)
            else:
                res["normalized_score"] = 1.0

        return results

    def retrieve(self, query, top_k=None, exclude_ids=None):
        """
        Retrieve documents using hybrid search (semantic + keyword).

        Args:
            query: The search query
            top_k: Number of documents to retrieve
            exclude_ids: Set of document IDs to exclude from results

        Returns:
            List of tuples (document_text, document_id)
        """
        if top_k is None:
            top_k = self.top_k

        if exclude_ids is None:
            exclude_ids = set()

        expanded_top_k = min(top_k * 3, 1000)

        # Semantic search (Pinecone)
        print(f"Performing semantic search for: {query[:50]}...")
        query_embedding = self._embed_query(query)

        pinecone_results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=expanded_top_k,
            include_values=False,
            include_metadata=True,
        )

        # Keyword search (Elasticsearch)
        print("Performing keyword search...")
        es_results = self.es_client.search(
            index=self.elasticsearch_index_name,
            body={
                "query": {"match": {"text": query}},
                "size": expanded_top_k
            }
        )

        # Parse Pinecone results
        semantic_results = []
        for match in pinecone_results.matches:
            semantic_results.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "score": match.score,
            })

        # Parse Elasticsearch results
        keyword_results = []
        for hit in es_results["hits"]["hits"]:
            keyword_results.append({
                "id": hit["_id"],
                "text": hit["_source"].get("text", ""),
                "score": hit["_score"],
            })

        # Normalize scores
        semantic_results = self._normalize_scores(semantic_results, "score")
        keyword_results = self._normalize_scores(keyword_results, "score")

        # Merge results
        combined = {}
        for res in semantic_results:
            if res["id"] not in exclude_ids:
                combined[res["id"]] = {
                    "id": res["id"],
                    "text": res["text"],
                    "semantic_score": res["normalized_score"],
                    "keyword_score": 0.0,
                }

        for res in keyword_results:
            if res["id"] not in exclude_ids:
                if res["id"] in combined:
                    combined[res["id"]]["keyword_score"] = res["normalized_score"]
                    # Use longer text if available
                    if len(res["text"]) > len(combined[res["id"]]["text"]):
                        combined[res["id"]]["text"] = res["text"]
                else:
                    combined[res["id"]] = {
                        "id": res["id"],
                        "text": res["text"],
                        "semantic_score": 0.0,
                        "keyword_score": res["normalized_score"],
                    }

        # Calculate hybrid scores
        for res in combined.values():
            res["final_score"] = (
                self.alpha * res["semantic_score"]
                + (1 - self.alpha) * res["keyword_score"]
            )

        # Sort and limit
        ranked_results = sorted(
            combined.values(), key=lambda x: x["final_score"], reverse=True
        )[:top_k]

        results = [(res["text"], res["id"]) for res in ranked_results]
        print(f"Retrieved {len(results)} documents using hybrid search")

        return results

    def get_stats(self):
        """Get statistics about indexed documents."""
        pinecone_stats = self.pinecone_index.describe_index_stats()
        es_count = self.es_client.count(index=self.elasticsearch_index_name)

        return {
            "pinecone_vectors": pinecone_stats.total_vector_count,
            "elasticsearch_docs": es_count["count"]
        }
