"""
Data processing pipeline: extract metadata and chunk case documents.
"""

import os
import re
from typing import List, Dict, Tuple
from pathlib import Path
from transformers import AutoTokenizer

# Initialize tokenizer for token-based chunking
_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def extract_metadata(content: str) -> Dict[str, str]:
    """
    Extract metadata from markdown front matter.
    Assumes metadata follows "## Metadata" header in markdown.
    Returns dict of key-value pairs.
    """
    metadata = {}
    
    # Find metadata section
    metadata_match = re.search(r'## Metadata\s*\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if not metadata_match:
        return metadata
    
    metadata_text = metadata_match.group(1)
    
    # Parse bullet points: "- **key:** value"
    for line in metadata_text.split('\n'):
        line = line.strip()
        if not line or not line.startswith('-'):
            continue
        
        # Remove bullet point marker
        line = line[1:].strip()
        
        # Parse "**key:** value" format
        match = re.match(r'\*\*([^*]+)\*\*:\s*(.+)', line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            metadata[key] = value
    
    return metadata


def chunk_document(content: str, lines_per_chunk: int = 300) -> List[str]:
    """
    Token-based chunking with overlap (surgical replacement).
    
    Note: lines_per_chunk parameter is retained for API compatibility but is not used.
    Actual chunking is controlled by token limits.
    
    Args:
        content: Document text to chunk
        lines_per_chunk: (Deprecated, retained for compatibility)
    
    Returns:
        List of text chunks
    """
    # Light semantic preprocessing: remove empty paragraphs
    paragraphs = content.split("\n\n")
    content = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    
    # Token-based chunking parameters
    chunk_size = 400      # tokens per chunk
    overlap = 80          # overlap tokens
    
    # Tokenize content
    tokens = _tokenizer.encode(content, add_special_tokens=False)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = _tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        start += chunk_size - overlap
    
    return chunks


def process_case_files(
    cases_dir: str = "data/cases",
    lines_per_chunk: int = 300
) -> List[Dict]:
    """
    Process all case markdown files.
    
    Returns list of dicts with:
    - case_id: unique identifier
    - chunk_idx: index of chunk within case
    - text: chunk content
    - metadata: extracted metadata
    """
    results = []
    cases_path = Path(cases_dir)
    
    for case_file in sorted(cases_path.glob("*.md")):
        with open(case_file, 'r') as f:
            content = f.read()
        
        # Extract metadata
        metadata = extract_metadata(content)
        
        # Generate case ID from filename
        case_id = case_file.stem
        
        # Chunk document
        chunks = chunk_document(content, lines_per_chunk)
        
        # Create records
        for chunk_idx, chunk_text in enumerate(chunks):
            results.append({
                'case_id': case_id,
                'chunk_idx': chunk_idx,
                'text': chunk_text,
                'metadata': metadata,
            })
    
    return results


if __name__ == '__main__':
    docs = process_case_files()
    print(f"Processed {len(docs)} chunks")
    if docs:
        print(f"Sample: {docs[0]}")
