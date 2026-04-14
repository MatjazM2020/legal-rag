# Legal Decision Justification RAG System

A retrieval-augmented generation system for legal case analysis using Qdrant vector store and Markov Decision Processes.

## Overview

This system answers legal queries by:
1. **Vectorizing** case documents into semantic embeddings
2. **Retrieving** the most relevant cases from Qdrant
3. **Generating** justified responses with traced reasoning
4. **Returning** full justification traces (facts → interpretation → conclusion → sources)

**Tech Stack:**
- Python 3.8+
- Flask (REST API)
- Qdrant (vector database)
- Sentence Transformers (embeddings: `all-MiniLM-L6-v2`, 384-dim)
- OpenAI API (LLM generation)

---

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant

Run Qdrant vector database locally on port 6333:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

(Or install locally: https://qdrant.tech/documentation/install/)

### 3. Prepare Data

Place legal case markdown files in the `cases/` folder:
- Each file should be named: `case_name.md`
- Files are read recursively: `cases/**/*.md`

Example structure:
```
cases/
  a_v_the_united_kingdom.md
  chahal_v_the_united_kingdom.md
  ...
```

### 4. Vectorize & Ingest

```bash
python rag_ingest.py
```

This will:
- Parse case markdown files
- Extract metadata sections
- Chunk documents into ~400-token segments
- Generate embeddings using `all-MiniLM-L6-v2`
- Load into Qdrant collection `legal_cases`

Output example:
```
✓ Processed 148 document chunks from 41 cases
✓ Successfully ingested 148 chunks
Vector store ready!
```

### 5. Start API Server

```bash
python rag_api.py
```

Server runs at `http://localhost:5001`

---

## API Usage

### Submit Legal Query

```bash
curl -X POST http://localhost:5001/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "due process in algorithmic systems"}'
```

**Response:**
```json
{
  "query": "due process in algorithmic systems",
  "response": "Based on retrieved cases...",
  "conclusion": "Primary reference: ...",
  "state": "justified",
  "reward": 0.85,
  "trace": {
    "facts": ["[case_id] excerpt...", ...],
    "interpretation": "...",
    "conclusion": "...",
    "sources": [...]
  }
}
```

### View Justification Trace

```bash
curl "http://localhost:5001/api/justification?q=due+process+in+algorithmic+systems"
```

### Health Check

```bash
curl http://localhost:5001/health
```

---

## System Architecture

### Data Pipeline
- `rag_data_processor.py`: Extract metadata, chunk documents
- `rag_vector_store.py`: Qdrant client, vectorization, retrieval

### Query Resolution (MDP)
- `rag_mdp.py`: Markov Decision Process state machine
- State sequence: QUERIED → RETRIEVED → GENERATED → JUSTIFIED
- Reward model based on retrieval quality + generation completeness

### REST API
- `rag_api.py`: Flask server with endpoints
- `rag_ingest.py`: CLI for document ingestion

---

## File Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── rag_api.py                   # Flask REST API
├── rag_ingest.py                # Ingestion CLI
├── rag_data_processor.py        # Document processing
├── rag_vector_store.py          # Qdrant wrapper
├── rag_mdp.py                   # Query MDP
├── rag_llm.py                   # LLM generation
├── RAG_SYSTEM.md                # Detailed architecture
└── cases/                       # Legal case markdown files
    └── *.md
```

---

## Environment Variables

Create `.env` file for configuration:

```
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_api_key_here
```

---

## Key Concepts

### Vectorization
Documents are split into chunks and embedded using `all-MiniLM-L6-v2` (384-dimensional vectors). COSINE similarity is used for retrieval.

### MDP (Markov Decision Process)
Query resolution follows a deterministic state machine:
- **Retrieval Stage**: Fetch relevant cases (reward: relevance scores)
- **Generation Stage**: Create response with facts (reward: completeness)
- **Justification Stage**: Complete trace with conclusion (reward: 1.0 if successful)

### Justification Trace
Returns provenance of reasoning:
- **Facts**: Key excerpts from top-3 retrieved cases
- **Interpretation**: LLM response analyzing relevance
- **Conclusion**: Final decision with case reference
- **Sources**: Full metadata and relevance scores

---

## Extending the System

**Custom Reward Functions:**
```python
mdp = LegalQueryMDP()
mdp.initialize_query(query)
custom_reward = len(mdp.context.retrieved_docs) / 5.0
```

**Multiple Collections:**
Create separate Qdrant collections per case type:
```python
vector_store = QdrantVectorStore(collection_name="echr_cases")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused to Qdrant | Ensure Docker container is running: `docker ps` |
| "Collection not found" | Run `python rag_ingest.py` first |
| Empty response | Check case files are in `cases/` folder |
| LLM generation fails | Verify `OPENAI_API_KEY` in `.env` |

---

## See Also

- `RAG_SYSTEM.md`: Detailed architecture and design rationale
- `data/metadata.csv`: Case metadata reference
- `data/README.md`: Data format specification

---

## License

Internal project for legal case analysis.
