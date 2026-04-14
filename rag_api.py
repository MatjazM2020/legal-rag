"""
Legal Decision Justification System - REST API

Architecture:
1. Legal Researcher submits query
2. System: embeds query, retrieves docs, generates response
3. Researcher views justification trace (facts -> interpretation -> conclusion -> sources)

Uses MDP for query resolution state management.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

from rag_vector_store import QdrantVectorStore
from rag_mdp import LegalQueryMDP, MDPAction
from rag_data_processor import process_case_files
from rag_llm import generate_text

load_dotenv()

app = Flask(__name__)
CORS(app)

# Global state
vector_store: QdrantVectorStore = None
_ingest_lock = False


def initialize_rag():
    """Initialize RAG system on first request."""
    global vector_store, _ingest_lock
    
    if vector_store is not None:
        return
    
    print("Initializing RAG system...")
    vector_store = QdrantVectorStore(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name="legal_cases"
    )
    
    # Check if collection exists and has data
    try:
        collection_info = vector_store.client.get_collection("legal_cases")
        print(f"Collection exists: {collection_info.points_count} points")
    except:
        # Need to ingest
        if not _ingest_lock:
            _ingest_lock = True
            print("Ingesting case documents...")
            docs = process_case_files(lines_per_chunk=300)
            vector_store.ingest_documents(docs)
            _ingest_lock = False


def extract_facts_from_docs(docs: List[Dict]) -> List[str]:
    """
    Simple fact extraction from retrieved documents.
    Takes first few lines from each doc as key facts.
    """
    facts = []
    for i, doc in enumerate(docs[:3]):  # Use top 3 docs
        text = doc['text'][:200]  # First 200 chars as fact
        case_id = doc.get('case_id', 'unknown')
        facts.append(f"[{case_id}] {text}...")
    return facts


def generate_response(query: str, docs: List[Dict]) -> str:
    if not docs:
        return "No relevant documents found."

    context = "\n\n".join([
        f"Case: {doc.get('case_id', 'unknown')}\nText: {doc['text'][:500]}"
        for doc in docs[:3]
    ])

    prompt = f"""
You are a legal assistant.

Query:
{query}

Relevant case excerpts:
{context}

Task:
Summarize the legal relevance of these cases to the query.
Be concise and structured.
"""

    return generate_text(prompt)


def generate_conclusion(query: str, docs: List[Dict]) -> str:
    if not docs:
        return "Insufficient information to draw a conclusion."

    context = "\n\n".join([
        f"{doc.get('case_id', 'unknown')}: {doc['text'][:300]}"
        for doc in docs[:3]
    ])

    prompt = f"""
You are a legal decision assistant.

Query:
{query}

Cases:
{context}

Task:
Provide a short legal conclusion referencing the most relevant case.
"""

    return generate_text(prompt)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


@app.route('/api/query', methods=['POST'])
def submit_query():
    """
    Submit Legal Query Use Case
    
    Includes (always executed):
    1. Embed query
    2. Retrieve relevant documents
    3. Generate justified response
    
    Input: { "query": "legal question" }
    Output: { "query_id": "uuid", "response": "...", "trace": {...} }
    """
    initialize_rag()
    
    data = request.json
    query_text = data.get('query', '').strip()
    
    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400
    
    # Initialize MDP for this query
    mdp = LegalQueryMDP()
    mdp.initialize_query(query_text)
    
    try:
        # RETRIEVED state: embed query and retrieve documents
        retrieved_docs = vector_store.retrieve(query_text, limit=5)
        mdp.step(MDPAction.RETRIEVE, docs=retrieved_docs)
        
        # Populate state variables for MDP context
        mdp.context.state_variables = {
            "num_docs": len(retrieved_docs),
            "top_score": retrieved_docs[0]["score"] if retrieved_docs else 0,
            "query_length": len(query_text)
        }
        
        # GENERATED state: extract facts and generate response
        facts = extract_facts_from_docs(retrieved_docs)
        response_text = generate_response(query_text, retrieved_docs)
        mdp.step(MDPAction.GENERATE, response_text=response_text, facts=facts)
        
        # JUSTIFIED state: final conclusion and sources
        conclusion = generate_conclusion(query_text, retrieved_docs)
        sources = [
            {
                'case_id': doc['case_id'],
                'chunk_idx': doc['chunk_idx'],
                'relevance': doc['score']
            }
            for doc in retrieved_docs
        ]
        mdp.step(MDPAction.JUSTIFY, conclusion=conclusion, sources=sources)
        
        # Return response
        return jsonify({
            'query': query_text,
            'response': response_text,
            'conclusion': conclusion,
            'state': mdp.current_state.value,
            'reward': mdp.get_total_reward(),
            'trace': mdp.get_trace()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/justification/<query_id>', methods=['GET'])
def view_justification(query_id):
    """
    View Justification Trace Use Case
    
    Includes (always executed):
    - Display facts
    - Display interpretation  
    - Display conclusion
    - Show cited sources with links
    
    This endpoint is called after submit_query to retrieve the full trace.
    For simplicity, we re-execute the query (in production, would store traces).
    """
    # In a real system, this would retrieve a stored trace by ID
    # For this minimal implementation, we accept query as parameter
    query_text = request.args.get('q', '').strip()
    
    if not query_text:
        return jsonify({'error': 'Query parameter required'}), 400
    
    initialize_rag()
    
    try:
        # Re-run the query to generate trace
        retrieved_docs = vector_store.retrieve(query_text, limit=5)
        facts = extract_facts_from_docs(retrieved_docs)
        response_text = generate_response(query_text, retrieved_docs)
        conclusion = generate_conclusion(query_text, retrieved_docs)
        sources = [
            {
                'case_id': doc['case_id'],
                'chunk_idx': doc['chunk_idx'],
                'relevance': doc['score'],
                'text': doc['text'][:300] + "..."
            }
            for doc in retrieved_docs
        ]
        
        return jsonify({
            'query': query_text,
            'justification_trace': {
                'facts': facts,
                'interpretation': response_text,
                'conclusion': conclusion,
                'sources': sources
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ingest', methods=['POST'])
def ingest_documents():
    """
    Admin endpoint: manually trigger document ingestion.
    """
    initialize_rag()
    
    try:
        docs = process_case_files(lines_per_chunk=300)
        count = vector_store.ingest_documents(docs)
        
        return jsonify({
            'status': 'success',
            'documents_ingested': count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
