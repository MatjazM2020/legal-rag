"""
Qdrant vector store connector and ingest pipeline.
"""

from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import hashlib


class QdrantVectorStore:
    """Minimal Qdrant-based vector store for legal documents."""
    
    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "legal_cases"):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
    
    def create_collection(self):
        """Create vector collection if it doesn't exist."""
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
    
    def ingest_documents(self, documents: List[Dict]) -> int:
        """
        Ingest documents into vector store.
        
        Args:
            documents: List of dicts with 'case_id', 'chunk_idx', 'text', 'metadata'
        
        Returns:
            Number of documents ingested
        """
        self.create_collection()
        
        # Batch encode all texts for efficiency
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_model.encode(texts, batch_size=32)
        
        points = []
        for doc, embedding in zip(documents, embeddings):
            # Generate unique ID based on case_id and chunk_idx
            doc_id = int(hashlib.md5(
                f"{doc['case_id']}#{doc['chunk_idx']}".encode()
            ).hexdigest()[:8], 16)
            
            # Prepare payload (metadata)
            payload = {
                'case_id': doc['case_id'],
                'chunk_idx': doc['chunk_idx'],
                'text': doc['text'],
            }
            # Add metadata fields
            for key, value in doc['metadata'].items():
                payload[f'meta_{key}'] = value
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload=payload
            ))
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Ingested {len(points)} document chunks")
        return len(points)
    
    def retrieve(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Returns list of dicts with 'text', 'metadata', 'score'
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query points
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit
        )
        
        # Format results
        docs = []
        for hit in results.points:
            doc = {
                'text': hit.payload.get('text', ''),
                'case_id': hit.payload.get('case_id', ''),
                'chunk_idx': hit.payload.get('chunk_idx', 0),
                'score': hit.score,
                'metadata': {}
            }
            
            # Extract metadata fields (those starting with 'meta_')
            for key, value in hit.payload.items():
                if key.startswith('meta_'):
                    doc['metadata'][key[5:]] = value
            
            docs.append(doc)
        
        return docs
    
    def clear_collection(self):
        """Delete collection (for cleanup)."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
        except:
            pass
