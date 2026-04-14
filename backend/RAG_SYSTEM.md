# Legal Decision Justification RAG System

A minimal Retrieval-Augmented Generation (RAG) system for legal case analysis using Qdrant vector store and Markov Decision Processes.

## Architecture

### System Components

1. **Data Pipeline** (`rag_data_processor.py`)
   - Extracts metadata from case markdown files
   - Chunks documents into ~300-line segments
   - Preserves metadata context across chunks

2. **Vector Store** (`rag_vector_store.py`)
   - Qdrant client wrapper
   - Embeds documents using `all-MiniLM-L6-v2` (384-dim)
   - COSINE similarity for retrieval

3. **Markov Decision Process** (`rag_mdp.py`)
   - States: QUERIED → RETRIEVED → GENERATED → JUSTIFIED
   - Deterministic state transitions
   - Reward model based on retrieval quality and generation completeness

4. **REST API** (`rag_api.py`)
   - Legal Researcher submits queries
   - System generates justified responses
   - View full justification traces with sources

### Use Cases

#### 1. Submit Legal Query
**Initiated by:** Legal Researcher

**Always executed:**
- Embed query using sentence transformer
- Retrieve top-5 relevant documents from Qdrant
- Generate justified response combining facts and interpretation

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "due process challenges in algorithmic sentencing"}'
```

**Response:**
```json
{
  "query": "due process challenges in algorithmic sentencing",
  "response": "Based on the query... [retrieved cases summary]",
  "conclusion": "Primary Decision Reference: State v. Loomis...",
  "state": "justified",
  "reward": 0.85,
  "trace": {
    "query": "...",
    "facts": ["[case_id] fact excerpt...", ...],
    "interpretation": "...",
    "conclusion": "...",
    "sources": [...]
  }
}
```

#### 2. View Justification Trace
**Accessed by:** Legal Researcher, Judge/Decision-maker

**Always executed:**
- Display facts extracted from topmost relevant documents
- Display interpretation of query against retrieved cases
- Display conclusion with decision reference
- Show cited sources with relevance scores

```bash
curl "http://localhost:5000/api/justification/q?q=due+process+challenges+in+algorithmic+sentencing"
```

## Markov Decision Process (MDP)

### State Space

```
S ∈ {QUERIED, RETRIEVED, GENERATED, JUSTIFIED}
```

| State | Meaning | Context |
|-------|---------|---------|
| QUERIED | Query received, initial state | Query text available |
| RETRIEVED | Documents retrieved from vector store | Query + top-5 retrieved docs |
| GENERATED | Response generated with facts | Query + docs + extracted facts |
| JUSTIFIED | Complete justification trace | Query + docs + facts + conclusion + sources |

### State Transitions

```
QUERIED --[retrieve]--> RETRIEVED --[generate]--> GENERATED --[justify]--> JUSTIFIED
   ↑                                                                             ↓
   └─────────────────────── Terminal State (is_terminal=True) ────────────────┘
```

Each transition is deterministic (given successful execution).

### Reward Model

The MDP accumulates reward across transitions:

- **Retrieval Reward** R_ret = avg(relevance_scores) of top-5 docs
- **Generation Reward** R_gen = 1.0 if facts extracted and response generated
- **Justification Reward** R_just = 1.0 if conclusion and sources provided

**Total Reward:**
```
R_total = (R_ret + R_gen + R_just) / 3
```

Range: [0, 1] where 1.0 = optimal response

### Why This Is a Valid MDP

1. **State Space (S):** Discrete, finite set of query states
2. **Action Space (A):** Implicit in transitions (retrieve, generate, justify)
3. **Transition Dynamics (T):** P(s'|s,a) = 1 for valid transitions (deterministic)
4. **Reward Function (R):** Well-defined based on component quality metrics
5. **Markov Property:** Future states depend only on current state and action, not history

## Setup & Usage

### Prerequisites

- Qdrant running locally: `localhost:6333`
- Python 3.8+

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

1. **Ingest documents:**
```bash
python rag_ingest.py
```

2. **Start API server:**
```bash
python rag_api.py
```

3. **Submit query:**
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "European human rights derogation in emergency"}'
```

## File Structure

```
.
├── rag_data_processor.py    # Extract metadata, chunk documents
├── rag_vector_store.py      # Qdrant wrapper, retrieval logic
├── rag_mdp.py               # MDP state machine & rewards
├── rag_api.py               # REST API (Flask)
├── rag_ingest.py            # CLI for document ingestion
└── data/cases/              # Case markdown files
    └── [case files]
```

## Design Rationale

### Minimalism

- **Single embedding model**: `all-MiniLM-L6-v2` (fast, sufficient for legal text)
- **Fixed chunk size**: 300 lines (covers most subcases)
- **Top-5 retrieval**: Balances relevance with diversity
- **Simple fact extraction**: First N characters of top documents

### MDP Justification

Rather than arbitrary state management, the MDP formalizes query resolution as a sequence of decisions:

1. **State encodes information**: Each state represents available knowledge post-action
2. **Transitions are decisions**: Retrieve, generate, justify are distinct decision points
3. **Rewards are measurable**: Relevance scores and generation quality are quantifiable
4. **Terminal condition**: Full justification achieved (Markov property holds)

This allows future extension to:
- **Reinforcement Learning**: Learn to choose actions adaptively
- **Policy Optimization**: Refine retrieve/generate parameters based on rewards
- **Multi-step Reasoning**: Chain multiple queries as MDPs

## Extending the System

### Add Custom Reward Functions

```python
mdp = LegalQueryMDP()
mdp.initialize_query(query)
mdp.transition_to_retrieved(docs)

# Override reward calculation
custom_reward = len(mdp.context.retrieved_docs) / 5.0
```

### Integration with Decision Systems

The justification trace can feed into decision-making systems:

```python
response = requests.post('http://localhost:5000/api/query', 
                        json={'query': 'my question'})
trace = response.json()['trace']

# Pass to decision-maker
judge_decision = make_decision(trace['facts'], trace['conclusion'])
```

### Scaling Considerations

- **Multiple collections**: Create separate Qdrant collections per case type
- **Batch queries**: Process multiple queries asynchronously
- **Cache traces**: Store MDP execution traces in persistent storage
- **Distributed embeddings**: Use larger model on GPU for retrieval quality

## API Reference

### POST /api/query

Submit a legal query and return full justification.

**Request:**
```json
{
  "query": "string"
}
```

**Response:**
```json
{
  "query": "string",
  "response": "string",
  "conclusion": "string",
  "state": "justified",
  "reward": 0.0-1.0,
  "trace": { ... }
}
```

### GET /api/justification/<query_id>

View justification trace for a query.

**Query Parameters:**
- `q`: The legal query string

**Response:**
```json
{
  "query": "string",
  "justification_trace": {
    "facts": ["string", ...],
    "interpretation": "string",
    "conclusion": "string",
    "sources": [
      {
        "case_id": "string",
        "chunk_idx": int,
        "relevance": float,
        "text": "string"
      }
    ]
  }
}
```

### GET /health

Health check endpoint.

## Notes for Future Enhancement

The system is intentionally minimal to satisfy MDP requirements. Potential enhancements:

- **Context window expansion**: Retrieve multiple chunks per case
- **Hierarchical retrieval**: First retrieve case → then chunk within case
- **Multi-agent modeling**: Separate MDPs for different legal actors (defense, prosecution, judge)
- **Policy learning**: Use case outcomes to optimize retrieval/generation policies
- **Explainability**: Log MDP state transitions and reward calculations
