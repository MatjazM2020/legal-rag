"""
Minimal Markov Decision Process for legal query resolution.

This implements a simple sequential decision process with states:
- S0: QUERIED (initial) - query received
- S1: RETRIEVED - documents retrieved from vector store
- S2: GENERATED - response generated with facts
- S3: JUSTIFIED - justification trace complete

Transitions: QUERIED -> RETRIEVED -> GENERATED -> JUSTIFIED
Each state encodes the query context and retrieval quality.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field


class MDPState(Enum):
    """Valid states in the legal query MDP."""
    QUERIED = "queried"       # Query received
    RETRIEVED = "retrieved"   # Documents retrieved
    GENERATED = "generated"   # Response generated
    JUSTIFIED = "justified"   # Trace complete


class MDPAction(Enum):
    """Valid actions in the legal query MDP."""
    RETRIEVE = "retrieve"     # Retrieve documents from vector store
    GENERATE = "generate"     # Generate response with facts
    JUSTIFY = "justify"       # Complete justification with conclusion


@dataclass
class MDPContext:
    """Context information shared throughout MDP execution."""
    query: str
    retrieved_docs: List[Dict[str, Any]] = None
    response_text: str = ""
    facts: List[str] = None
    conclusion: str = ""
    sources: List[Dict] = None
    state_variables: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.retrieved_docs is None:
            self.retrieved_docs = []
        if self.facts is None:
            self.facts = []
        if self.sources is None:
            self.sources = []


class LegalQueryMDP:
    """
    Minimal MDP for legal query resolution.
    
    Implements state transitions and simple reward model:
    - Reward for retrieval quality (document relevance)
    - Reward for response generation
    - Terminal reward for complete justification
    """
    
    def __init__(self):
        self.current_state = MDPState.QUERIED
        self.context: Optional[MDPContext] = None
        self.state_history: List[MDPState] = []
    
    def initialize_query(self, query: str) -> MDPContext:
        """Initialize a new query in QUERIED state."""
        self.current_state = MDPState.QUERIED
        self.context = MDPContext(query=query)
        self.state_history = [MDPState.QUERIED]
        return self.context
    
    def step(self, action: MDPAction, **kwargs) -> MDPContext:
        """
        Execute a single MDP step: state transition with action.
        
        Args:
            action: MDPAction to execute (RETRIEVE, GENERATE, or JUSTIFY)
            **kwargs: Action-specific parameters
                - RETRIEVE: docs (List[Dict])
                - GENERATE: response_text, facts
                - JUSTIFY: conclusion, sources
        
        Returns:
            Updated MDPContext
        
        Raises:
            ValueError: Invalid state-action pair
        """
        if self.current_state == MDPState.QUERIED and action == MDPAction.RETRIEVE:
            # QUERIED + RETRIEVE -> RETRIEVED
            self.context.retrieved_docs = kwargs.get("docs", [])
            self.current_state = MDPState.RETRIEVED
        
        elif self.current_state == MDPState.RETRIEVED and action == MDPAction.GENERATE:
            # RETRIEVED + GENERATE -> GENERATED
            self.context.response_text = kwargs.get("response_text", "")
            self.context.facts = kwargs.get("facts", [])
            self.current_state = MDPState.GENERATED
        
        elif self.current_state == MDPState.GENERATED and action == MDPAction.JUSTIFY:
            # GENERATED + JUSTIFY -> JUSTIFIED
            self.context.conclusion = kwargs.get("conclusion", "")
            self.context.sources = kwargs.get("sources", [])
            self.current_state = MDPState.JUSTIFIED
        
        else:
            raise ValueError(f"Invalid transition: {self.current_state.value} + {action.value}")
        
        self.state_history.append(self.current_state)
        return self.context
    
    def get_retrieval_reward(self) -> float:
        """
        Compute reward for retrieval quality.
        Based on: average relevance score of retrieved documents.
        Proxy for legal case relevance to the query.
        Range: [0, 1]
        """
        if not self.context.retrieved_docs:
            return 0.0
        
        avg_score = sum(d.get('score', 0) for d in self.context.retrieved_docs) / len(self.context.retrieved_docs)
        return min(avg_score, 1.0)
    
    def get_generation_reward(self) -> float:
        """
        Compute reward for response generation.
        Based on: whether facts were extracted and response generated.
        Range: [0, 1]
        """
        if not self.context.response_text:
            return 0.0
        if not self.context.facts:
            return 0.5
        return 1.0
    
    def get_justification_reward(self) -> float:
        """
        Compute terminal reward for complete justification.
        Based on: whether conclusion and sources are provided.
        Range: [0, 1]
        """
        if not self.context.conclusion or not self.context.sources:
            return 0.0
        return 1.0
    
    def get_total_reward(self) -> float:
        """
        Compute total cumulative reward across all transitions.
        Average of rewards from states reached in history.
        """
        retrieval_r = self.get_retrieval_reward() if MDPState.RETRIEVED in self.state_history else 0
        generation_r = self.get_generation_reward() if MDPState.GENERATED in self.state_history else 0
        justification_r = self.get_justification_reward() if MDPState.JUSTIFIED in self.state_history else 0
        
        return (retrieval_r + generation_r + justification_r) / 3.0
    
    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self.current_state == MDPState.JUSTIFIED
    
    def get_trace(self) -> Dict[str, Any]:
        """Get full justification trace."""
        return {
            'query': self.context.query,
            'facts': self.context.facts,
            'interpretation': self.context.response_text,
            'conclusion': self.context.conclusion,
            'sources': self.context.sources,
            'state_history': [s.value for s in self.state_history],
            'total_reward': self.get_total_reward()
        }
