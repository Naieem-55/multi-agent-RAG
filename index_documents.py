"""
Document Indexing Script
Indexes documents into Pinecone and Elasticsearch for the local hybrid retriever.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Index documents for RAGentA")
    parser.add_argument(
        "--documents",
        type=str,
        default="sample_documents.json",
        help="Path to JSON file with documents (list of {id, text} objects)"
    )
    parser.add_argument(
        "--pinecone-api-key",
        type=str,
        default=None,
        help="Pinecone API key (or set PINECONE_API_KEY env var)"
    )
    parser.add_argument(
        "--pinecone-index",
        type=str,
        default="ragenta-index",
        help="Pinecone index name"
    )
    parser.add_argument(
        "--elasticsearch-url",
        type=str,
        default="http://localhost:9200",
        help="Elasticsearch URL"
    )
    parser.add_argument(
        "--elasticsearch-index",
        type=str,
        default="ragenta-docs",
        help="Elasticsearch index name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for indexing"
    )
    args = parser.parse_args()

    # Load documents
    print(f"Loading documents from {args.documents}...")
    with open(args.documents, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents")

    # Initialize retriever (this also creates indexes if needed)
    from local_hybrid_retriever import LocalHybridRetriever

    retriever = LocalHybridRetriever(
        pinecone_api_key=args.pinecone_api_key,
        pinecone_index_name=args.pinecone_index,
        elasticsearch_url=args.elasticsearch_url,
        elasticsearch_index_name=args.elasticsearch_index,
    )

    # Index documents
    retriever.index_documents(documents, batch_size=args.batch_size)

    # Print stats
    stats = retriever.get_stats()
    print(f"\nIndexing complete!")
    print(f"Pinecone vectors: {stats['pinecone_vectors']}")
    print(f"Elasticsearch docs: {stats['elasticsearch_docs']}")


if __name__ == "__main__":
    main()
