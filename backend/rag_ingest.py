#!/usr/bin/env python3
"""
Ingest legal case documents into Qdrant vector store.

Usage:
    python rag_ingest.py
"""

from rag_data_processor import process_case_files
from rag_vector_store import QdrantVectorStore


def main():
    print("Processing case documents...")
    docs = process_case_files(lines_per_chunk=300)
    print(f"✓ Processed {len(docs)} document chunks from {len(set(d['case_id'] for d in docs))} cases")
    
    print("\nConnecting to Qdrant...")
    vector_store = QdrantVectorStore(
        url="http://localhost:6333",
        collection_name="legal_cases"
    )
    
    print(f"Ingesting documents into Qdrant...")
    count = vector_store.ingest_documents(docs)
    print(f"✓ Successfully ingested {count} chunks")
    
    print("\nVector store ready!")
    print(f"Collection: legal_cases")
    print(f"Total points: {count}")


if __name__ == '__main__':
    main()
