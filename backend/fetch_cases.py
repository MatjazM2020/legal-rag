"""
ECHR Case Fetcher & Markdown Generator
=======================================
Fetches ECtHR cases from HUDOC and converts them to structured markdown files
for use in a RAG (Retrieval-Augmented Generation) pipeline.

Usage:
    # Fetch pilot cases only (marked with [PILOT] in case_list.txt)
    python fetch_cases.py --pilot

    # Fetch all 50 cases
    python fetch_cases.py --all

    # Fetch a specific case by name
    python fetch_cases.py --case "Salduz v. Turkey"

    # Fetch cases by number range (e.g., first 10)
    python fetch_cases.py --range 1 10
"""

import re
import json
import csv
import time
import argparse
import requests
import xml.etree.ElementTree as ET
from io import BytesIO
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
import unicodedata
from pypdf import PdfReader


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data/cases"
RAW_DIR = BASE_DIR / "data/raw"
METADATA_FILE = BASE_DIR / "data/metadata/metadata.csv"
FAILED_MANIFEST_FILE = BASE_DIR / "data/raw/failed_cases_manifest.json"
HUDOC_SEARCH_URL = "https://hudoc.echr.coe.int/app/transform/rss"
REQUEST_DELAY = 2  # seconds between requests (be polite to HUDOC)

# Explicit aliases for a handful of known HUDOC title/indexing quirks.
CASE_SEARCH_ALIASES = {
    "bankovic_and_others_v_belgium_and_others": [
        "bankovic and others",
        "bankovic",
        "52207/99",
    ],
    "ohalloran_and_francis_v_the_united_kingdom": [
        "ohalloran and francis",
        "o halloran and francis",
        "15809/02",
        "25624/02",
    ],
    "navone_and_others_v_monaco": [
        "navone and others",
        "navone monaco",
        "62880/11",
        "62892/11",
        "62899/11",
    ],
    "ma_v_france": [
        "m a and others france",
        "m.a. and others france",
        "9373/15",
    ],
}

# ECHR Articles reference (Arts 1-19)
ECHR_ARTICLES = {
    "1": "Obligation to respect human rights",
    "2": "Right to life",
    "3": "Prohibition of torture",
    "4": "Prohibition of slavery and forced labour",
    "5": "Right to liberty and security",
    "6": "Right to a fair trial",
    "7": "No punishment without law",
    "8": "Right to respect for private and family life",
    "9": "Freedom of thought, conscience and religion",
    "10": "Freedom of expression",
    "11": "Freedom of assembly and association",
    "12": "Right to marry",
    "13": "Right to an effective remedy",
    "14": "Prohibition of discrimination",
    "15": "Derogation in time of emergency",
    "16": "Restrictions on political activity of aliens",
    "17": "Prohibition of abuse of rights",
    "18": "Limitation on use of restrictions on rights",
    "19": "Establishment of the Court",
}


# =============================================================================
# HUDOC API INTERACTION
# =============================================================================

def normalize_search_title(case_title: str) -> str:
    """Normalize a case title into HUDOC-friendly search text."""
    text = case_title.strip().lower()
    text = text.replace("v.", "v")
    text = re.sub(r"[’'\".,:;()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_case_key(text: str) -> str:
    """Normalize case names and metadata keys into a shared lookup form."""
    if not text:
        return ""

    replacements = str.maketrans({
        "ß": "ss",
        "æ": "ae",
        "œ": "oe",
        "ø": "o",
        "ł": "l",
        "đ": "d",
        "ð": "d",
        "þ": "th",
        "ı": "i",
        "ĳ": "ij",
    })

    normalized = unicodedata.normalize("NFKD", text)
    normalized = normalized.translate(replacements)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.casefold()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def extract_item_id(link: str) -> str | None:
    """Extract the HUDOC item ID from an RSS item link."""
    match = re.search(r'"itemid":\["([^"]+)"\]', link)
    return match.group(1) if match else None


def build_hudoc_query(search_text: str, strict_collections: bool = True) -> str:
    """Build the HUDOC query in strict (judgments) or relaxed mode."""
    query = (
        'contentsitename:ECHR AND (NOT (doctype=PR OR doctype=HFCOMOLD OR doctype=HECOMOLD)) '
        f'AND ({search_text})'
    )
    if strict_collections:
        query += ' AND ((documentcollectionid="GRANDCHAMBER") OR (documentcollectionid="CHAMBER"))'
    return query


def fetch_hudoc_rss(search_text: str, strict_collections: bool = True) -> list[dict]:
    """Fetch HUDOC RSS search results for a given search text."""
    params = {
        "library": "echreng",
        "query": build_hudoc_query(search_text, strict_collections=strict_collections),
        "sort": "",
        "start": 0,
        "rankingModelId": "22222222-eeee-0000-0000-000000000000",
    }

    headers = {
        "Accept": "application/rss+xml, application/xml, text/xml",
        "User-Agent": "Mozilla/5.0 (academic research project)",
    }

    response = requests.get(HUDOC_SEARCH_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    results = []
    for item in root.findall("./channel/item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        item_id = extract_item_id(link)
        if item_id:
            results.append({"title": title, "link": link, "description": description, "itemid": item_id})

    return results


def add_variant(variants: list[str], variant: str):
    """Append a search variant if it is non-empty and not already present."""
    normalized = variant.strip()
    if normalized and normalized not in variants:
        variants.append(normalized)


def build_search_variants(case_title: str, metadata: dict | None = None) -> list[str]:
    """Build deterministic query variants for HUDOC lookup."""
    variants: list[str] = []

    add_variant(variants, normalize_search_title(case_title))

    simple_name = case_title.split(" v. ")[0].strip() if " v. " in case_title else case_title.strip()
    add_variant(variants, normalize_search_title(simple_name))

    simple_tokens = normalize_search_title(simple_name).split()
    if simple_tokens:
        add_variant(variants, simple_tokens[0])
        if len(simple_tokens) > 1:
            add_variant(variants, " ".join(simple_tokens[:2]))

    normalized_case_key = normalize_case_key(case_title)
    for alias in CASE_SEARCH_ALIASES.get(normalized_case_key, []):
        add_variant(variants, alias)

    if metadata:
        file_id_query = str(metadata.get("file_id", "")).replace("_", " ").strip()
        add_variant(variants, file_id_query)

        app_no = metadata.get("application_number", "")
        if app_no:
            for part in re.split(r"[;,]", str(app_no)):
                part = part.strip()
                if part:
                    add_variant(variants, part)
                    if re.search(r"\d+\/\d+", part):
                        add_variant(variants, f'appno:"{part}"')

    safe_variants: list[str] = []
    for variant in variants:
        # HUDOC returns 400 for certain very short text variants like "m a".
        no_space = re.sub(r"\s+", "", variant)
        if len(no_space) < 3 and not any(char.isdigit() for char in no_space):
            continue
        add_variant(safe_variants, variant)

    return safe_variants


def tokenized_case_title(text: str) -> list[str]:
    """Tokenize a title for overlap-based result scoring."""
    tokens = [token for token in normalize_case_key(text).split("_") if token not in {"case", "of", "the", "and", "v"}]
    return tokens


def select_best_hudoc_result(case_title: str, results: list[dict], reference_title: str | None = None) -> dict | None:
    """Choose the best matching HUDOC result for the requested case title."""
    if not results:
        return None

    normalized_case_title = normalize_case_key(reference_title or case_title)
    reference_tokens = tokenized_case_title(reference_title or case_title)

    ranked_candidates: list[tuple[tuple[int, int, int, int, str], dict]] = []

    def score(result: dict) -> tuple[int, int, int, int, str]:
        title = normalize_case_key(result.get("title", ""))
        candidate_tokens = set(tokenized_case_title(result.get("title", "")))
        shared_tokens = len(set(reference_tokens) & candidate_tokens)
        exact = int(normalized_case_title in title)
        starts_case_of = int(title.startswith("case_of"))
        translation_penalty = int("translation" in title or "translated" in title)
        return (exact, shared_tokens, starts_case_of, -translation_penalty, title)

    for candidate in results:
        candidate_title_key = normalize_case_key(candidate.get("title", ""))
        if "translation" in candidate_title_key or "translated" in candidate_title_key:
            continue
        ranked_candidates.append((score(candidate), candidate))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[0], reverse=True)
    best = ranked_candidates[0][1]
    best_tokens = set(tokenized_case_title(best.get("title", "")))
    shared_tokens = len(set(reference_tokens) & best_tokens)
    overlap_ratio = shared_tokens / max(len(reference_tokens), 1)

    if overlap_ratio >= 0.75:
        return best

    return None


def search_hudoc(case_title: str, metadata: dict | None = None) -> dict | None:
    """Search HUDOC for a case by title and return metadata."""
    try:
        search_variants = build_search_variants(case_title, metadata=metadata)

        for strict_mode in (True, False):
            for search_text in search_variants:
                try:
                    results = fetch_hudoc_rss(search_text, strict_collections=strict_mode)
                except Exception as search_error:
                    print(f"  [!] HUDOC query failed for '{search_text}' ({'strict' if strict_mode else 'relaxed'}): {search_error}")
                    continue

                selected = select_best_hudoc_result(
                    case_title,
                    results,
                    reference_title=metadata.get("case_name") if metadata else None,
                )
                if selected:
                    return {
                        "itemid": selected["itemid"],
                        "docname": selected.get("title", case_title),
                        "respondent": "N/A",
                        "doctypebranch": "GRANDCHAMBER/CHAMBER",
                    }

        print(f"  [!] No results found for: {case_title}")
        return search_hudoc_simple(case_title, metadata=metadata)

    except Exception as e:
        print(f"  [ERROR] Search failed for '{case_title}': {e}")
        return None


def search_hudoc_simple(case_title: str, metadata: dict | None = None) -> dict | None:
    """Fallback: simpler search query if the strict search fails."""
    try:
        search_variants = build_search_variants(case_title, metadata=metadata)

        for search_text in search_variants:
            try:
                results = fetch_hudoc_rss(search_text, strict_collections=False)
            except Exception as search_error:
                print(f"  [!] Fallback HUDOC query failed for '{search_text}': {search_error}")
                continue

            selected = select_best_hudoc_result(
                case_title,
                results,
                reference_title=metadata.get("case_name") if metadata else None,
            )
            if selected:
                return {
                    "itemid": selected["itemid"],
                    "docname": selected.get("title", case_title),
                    "respondent": "N/A",
                    "doctypebranch": "GRANDCHAMBER/CHAMBER",
                }
        print(f"  [!] Fallback search also found nothing for: {case_title}")
        return None
    except Exception as e:
        print(f"  [ERROR] Fallback search failed: {e}")
        return None


def fetch_full_text(item_id: str) -> str | None:
    """
    Fetch the full HTML text of a judgment from HUDOC using item ID.
    """
    headers = {
        "Accept": "application/pdf, text/html;q=0.9, */*;q=0.8",
        "User-Agent": "Mozilla/5.0 (academic research project)",
    }

    urls = [
        f"https://hudoc.echr.coe.int/app/conversion/pdf/?library=ECHR&id={item_id}",
        f"https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id={item_id}",
        f"https://hudoc.echr.coe.int/eng?i={item_id}",
    ]

    last_error = None
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()

            if response.status_code == 204:
                continue

            if not response.content:
                continue

            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type or "/pdf/" in url:
                pdf_reader = PdfReader(BytesIO(response.content))
                pages = []
                for page in pdf_reader.pages:
                    pages.append(page.extract_text() or "")
                text = "\n".join(pages).strip()
                if text:
                    return text
                continue
            else:
                text = response.text.strip()
                if not text:
                    continue

                lower_text = text.lower()
                is_hudoc_shell = (
                    "<!doctype html" in lower_text
                    and "<title>hudoc - european court of human rights</title>" in lower_text
                    and "global_url_query = ''" in lower_text
                )
                if is_hudoc_shell:
                    continue

                return text
        except Exception as e:
            last_error = e

    print(f"  [ERROR] Failed to fetch full text for {item_id}: {last_error}")
    return None


# =============================================================================
# HTML PARSING — Extract sections from ECtHR judgment HTML
# =============================================================================

def parse_judgment_html(html_content: str) -> dict:
    """
    Parse an ECtHR judgment HTML into structured sections.

    ECtHR judgments typically follow this structure:
    - PROCEDURE section
    - THE FACTS / THE CIRCUMSTANCES OF THE CASE
    - RELEVANT LEGAL FRAMEWORK / RELEVANT DOMESTIC LAW
    - THE LAW (court's legal reasoning)
    - FOR THESE REASONS, THE COURT... (operative provisions / decision)
    - Separate opinions (if any)
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")

    # Clean up the text
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = "\n".join(lines)

    sections = {
        "procedure": "",
        "facts": "",
        "relevant_law": "",
        "court_reasoning": "",
        "decision": "",
        "separate_opinions": "",
        "full_text": full_text,
    }

    # Common section header patterns in ECtHR judgments
    section_patterns = [
        (r"(?i)\bPROCEDURE\b", "procedure"),
        (r"(?i)\b(?:THE\s+)?FACTS?\b", "facts"),
        (r"(?i)\b(?:THE\s+CIRCUMSTANCES\s+OF\s+THE\s+CASE|CIRCUMSTANCES\s+OF\s+THE\s+CASE)\b", "facts"),
        (r"(?i)\bRELEVANT\s+(?:LEGAL\s+FRAMEWORK|DOMESTIC\s+(?:LAW|LEGISLATION|PRACTICE)|LAW\s+AND\s+PRACTICE)\b", "relevant_law"),
        (r"(?i)\bTHE\s+LAW\b", "court_reasoning"),
        (r"(?i)\bFOR\s+THESE\s+REASONS,?\s*THE\s+COURT\b", "decision"),
        (r"(?i)\b(?:JOINT\s+)?(?:PARTLY\s+)?(?:CONCURRING|DISSENTING|SEPARATE)\s+OPINION\b", "separate_opinions"),
    ]

    # Find section boundaries
    boundaries = []
    for pattern, section_name in section_patterns:
        for match in re.finditer(pattern, full_text):
            boundaries.append((match.start(), section_name, match.group()))

    # Sort by position
    boundaries.sort(key=lambda x: x[0])

    # Extract text between boundaries
    for i, (start_pos, section_name, header_text) in enumerate(boundaries):
        # Find end: next boundary or end of text
        if i + 1 < len(boundaries):
            end_pos = boundaries[i + 1][0]
        else:
            end_pos = len(full_text)

        section_text = full_text[start_pos:end_pos].strip()
        # Remove the header itself from the section text
        section_text = section_text[len(header_text):].strip()

        # If we already have content for this section, append
        if sections[section_name]:
            sections[section_name] += "\n\n" + section_text
        else:
            sections[section_name] = section_text

    return sections


def extract_articles_from_text(text: str) -> list[str]:
    """Extract ECHR article references from text."""
    # Match patterns like "Article 3", "Art. 6", "Article 6 § 1"
    pattern = r"(?:Article|Art\.?)\s+(\d{1,2})(?:\s*§\s*\d+)?"
    matches = re.findall(pattern, text)
    # Deduplicate and filter to Arts 1-19
    articles = sorted(set(m for m in matches if 1 <= int(m) <= 19), key=int)
    return articles


def extract_violation_info(conclusion_field: str) -> tuple[list[str], list[str]]:
    """
    Parse the HUDOC 'conclusion' field to extract violation/no-violation info.
    Returns (violations, no_violations) as lists of article strings.
    """
    violations = []
    no_violations = []

    if not conclusion_field:
        return violations, no_violations

    parts = conclusion_field.split(";")
    for part in parts:
        part = part.strip().lower()
        art_matches = re.findall(r"art(?:icle)?\.?\s*(\d{1,2})", part)
        if "no violation" in part or "non-violation" in part:
            no_violations.extend(art_matches)
        elif "violation" in part:
            violations.extend(art_matches)

    return (
        sorted(set(v for v in violations if 1 <= int(v) <= 19), key=int),
        sorted(set(v for v in no_violations if 1 <= int(v) <= 19), key=int),
    )


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================

def generate_markdown(metadata: dict, sections: dict) -> str:
    """
    Generate a structured markdown file from case metadata and parsed sections.
    """
    # Extract fields from HUDOC metadata
    doc_name = metadata.get("docname") or metadata.get("case_name", "Unknown Case")
    app_no = metadata.get("application_number") or metadata.get("appno", "N/A")
    judgment_date = metadata.get("judgment_date") or metadata.get("judgmentdate", "")
    respondent = metadata.get("respondent_state") or metadata.get("respondent", "N/A")
    importance = metadata.get("importance", "N/A")
    branch = metadata.get("chamber") or metadata.get("doctypebranch", "N/A")
    ecli = metadata.get("ecli", "N/A") or "N/A"
    article_field = metadata.get("articles") or metadata.get("article", "")
    conclusion_field = metadata.get("conclusion", "")
    scl = metadata.get("keywords") or metadata.get("scl", "")
    sep_opinion = metadata.get("separate_opinion") or metadata.get("separateopinion", "")

    # Format date
    if judgment_date:
        try:
            dt = datetime.fromisoformat(judgment_date.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_str = judgment_date[:10] if len(judgment_date) >= 10 else judgment_date
    else:
        date_str = "N/A"

    # Parse articles and violations
    articles_cited = extract_articles_from_text(article_field or "")
    violations, no_violations = extract_violation_info(conclusion_field or "")

    # Build article citations with names
    article_entries = []
    for art in articles_cited:
        name = ECHR_ARTICLES.get(art, "")
        article_entries.append(f"Art. {art}" + (f" ({name})" if name else ""))

    # Build violation summary
    violation_parts = []
    if violations:
        violation_parts.append("Violation: " + ", ".join(f"Art. {v}" for v in violations))
    if no_violations:
        violation_parts.append("No violation: " + ", ".join(f"Art. {v}" for v in no_violations))
    violation_summary = " | ".join(violation_parts) if violation_parts else "See decision section"

    # Map importance levels
    importance_map = {"1": "Key case", "2": "Level 1", "3": "Level 2", "4": "Level 3"}
    importance_str = importance_map.get(str(importance), str(importance))

    # Clean up sections — fallback messages if empty
    facts = sections.get("facts", "").strip() or "_Facts section not automatically extracted. See full text below._"
    relevant_law = sections.get("relevant_law", "").strip()
    reasoning = sections.get("court_reasoning", "").strip() or "_Reasoning section not automatically extracted. See full text below._"
    decision = sections.get("decision", "").strip() or "_Decision section not automatically extracted. Search for 'FOR THESE REASONS' in full text._"
    separate_opinions = sections.get("separate_opinions", "").strip()

    separate_opinion_flag = str(sep_opinion).strip().lower() in {"yes", "true", "1", "y"}

    # Build markdown
    md = f"""# {doc_name}

## Metadata
- **Application Number:** {app_no}
- **ECLI:** {ecli}
- **Court:** European Court of Human Rights
- **Chamber:** {branch}
- **Date of Judgment:** {date_str}
- **Respondent State:** {respondent}
- **Importance:** {importance_str}
- **Articles Cited:** {', '.join(article_entries) if article_entries else 'N/A'}
- **Violation Found:** {violation_summary}
- **Separate Opinion:** {"Yes" if separate_opinion_flag else "No"}
- **Keywords:** {scl if scl else 'N/A'}

## Facts

{facts}

## Relevant Legal Framework

{relevant_law if relevant_law else '_No separate legal framework section identified._'}

## Court's Reasoning

{reasoning}

## Decision

{decision}
"""

    if separate_opinions:
        md += f"""
## Separate Opinions

{separate_opinions}
"""

    # Add significance note (empty for manual completion)
    md += """
## Significance

_TODO: Add a brief note on why this case is a landmark/important precedent._

---
*Auto-generated from HUDOC. Review and verify all sections for accuracy.*
"""

    return md


# =============================================================================
# FILE I/O
# =============================================================================


def load_metadata(filepath: Path) -> dict:
    """Load metadata CSV into a dict keyed by file_id."""
    metadata = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys = []
            for field in ("file_id", "case_name", "title", "case_title"):
                value = row.get(field, "")
                key = normalize_case_key(value)
                if key and key not in keys:
                    keys.append(key)
            for key in keys:
                metadata.setdefault(key, row)
    return metadata

def load_case_list(filepath: str, pilot_only: bool = False) -> list[str]:
    """Load case names from the case list file."""
    cases = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            is_pilot = "[PILOT]" in line
            name = line.replace("[PILOT]", "").strip()
            if pilot_only and not is_pilot:
                continue
            cases.append(name)
    return cases


def sanitize_filename(name: str) -> str:
    """Convert a case name to a safe filename."""
    # e.g., "Salduz v. Turkey" -> "salduz_v_turkey"
    return normalize_case_key(name)


def save_raw_data(item_id: str, metadata: dict, html: str, case_name: str):
    """Save raw HUDOC data for reproducibility."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(case_name)

    with open(RAW_DIR / f"{safe_name}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    with open(RAW_DIR / f"{safe_name}_judgment.html", "w", encoding="utf-8") as f:
        f.write(html)


def resolve_case_file_id(case_name: str) -> str:
    """Convert a case name to the file_id used by metadata.csv."""
    return normalize_case_key(case_name)


def find_case_metadata(metadata_rows: dict, case_name: str) -> tuple[dict | None, str | None]:
    """Resolve the best metadata row for a case name and report the matched key."""
    keys = []
    primary = normalize_case_key(case_name)
    if primary:
        keys.append(primary)

    if " v. " in case_name:
        title_only = case_name.split(" v. ", 1)[0].strip()
        title_key = normalize_case_key(title_only)
        if title_key and title_key not in keys:
            keys.append(title_key)

    for key in keys:
        if key in metadata_rows:
            return metadata_rows[key], key

    return None, None


def load_failed_manifest() -> dict | None:
    """Load the most recent failed-case manifest if it exists."""
    if not FAILED_MANIFEST_FILE.exists():
        return None
    try:
        with open(FAILED_MANIFEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_failed_manifest(processed_cases: list[str], failed_cases: list[str], mode: str):
    """Persist failed-case information for retry-focused runs."""
    FAILED_MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "processed": processed_cases,
        "failed": failed_cases,
        "success_count": len(processed_cases) - len(failed_cases),
    }
    with open(FAILED_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def markdown_exists_for_case(case_name: str) -> bool:
    """Return True if markdown output already exists for a case name."""
    safe_name = sanitize_filename(case_name)
    return (OUTPUT_DIR / f"{safe_name}.md").exists()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_case(case_name: str, metadata_rows: dict) -> bool:
    """
    Full pipeline for a single case:
    1. Search HUDOC for the case
    2. Fetch full judgment text
    3. Parse into sections
    4. Generate markdown
    5. Save files
    """
    print(f"\n{'='*60}")
    print(f"Processing: {case_name}")
    print(f"{'='*60}")

    file_id = resolve_case_file_id(case_name)
    case_meta, metadata_key = find_case_metadata(metadata_rows, case_name)
    if not case_meta:
        print(f"  [SKIP] No metadata row found for: {case_name} ({file_id})")
        return False
    if metadata_key and metadata_key != file_id:
        print(f"  Metadata match: {metadata_key}")

    # Step 1: Search HUDOC
    print("  [1/5] Searching HUDOC...")
    hudoc_metadata = search_hudoc(case_name, metadata=case_meta)
    if not hudoc_metadata:
        print(f"  [SKIP] Could not find case on HUDOC: {case_name}")
        return False

    item_id = hudoc_metadata.get("itemid")
    doc_name = hudoc_metadata.get("docname", case_name)
    print(f"  Found: {doc_name} (ID: {item_id})")

    metadata = dict(case_meta)
    metadata["itemid"] = item_id
    metadata["docname"] = doc_name

    time.sleep(REQUEST_DELAY)

    # Step 2: Fetch full text
    print("  [2/5] Fetching full judgment text...")
    html_content = fetch_full_text(item_id)
    if not html_content:
        print(f"  [SKIP] Could not fetch judgment text for: {case_name}")
        return False

    time.sleep(REQUEST_DELAY)

    # Step 3: Save raw data
    print("  [3/5] Saving raw data...")
    save_raw_data(item_id, metadata, html_content, case_name)

    # Step 4: Parse sections
    print("  [4/5] Parsing judgment sections...")
    sections = parse_judgment_html(html_content)

    # Report what was found
    for section_name, content in sections.items():
        if section_name == "full_text":
            continue
        status = "✓" if content else "✗"
        length = len(content) if content else 0
        print(f"       {status} {section_name}: {length} chars")

    # Step 5: Generate and save markdown
    print("  [5/5] Generating markdown...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(case_name)
    md_content = generate_markdown(metadata, sections)

    md_path = OUTPUT_DIR / f"{safe_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"  [DONE] Saved to {md_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ECtHR cases from HUDOC and convert to markdown"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Fetch only pilot cases (5 cases)")
    group.add_argument("--all", action="store_true", help="Fetch all 50 cases")
    group.add_argument("--case", type=str, help="Fetch a single case by name")
    group.add_argument("--retry-failed", action="store_true",
                       help="Retry only failed cases from the previous run")
    group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                       help="Fetch cases by line number range (1-indexed)")

    parser.add_argument("--case-list", default=BASE_DIR / "cases/case_list.txt",
                        help="Path to case list file (default: cases/case_list.txt)")
    parser.add_argument("--include-existing", action="store_true",
                        help="With --retry-failed, include failed cases even if markdown already exists")

    args = parser.parse_args()

    print("=" * 60)
    print("ECHR Case Fetcher & Markdown Generator")
    print("=" * 60)

    run_mode = "case"
    if args.case:
        cases = [args.case]
    elif args.retry_failed:
        run_mode = "retry-failed"
        manifest = load_failed_manifest()
        if manifest and manifest.get("failed"):
            failed_cases = [str(case).strip() for case in manifest.get("failed", []) if str(case).strip()]
        else:
            # Fallback: derive unresolved cases from missing markdown outputs.
            all_cases = load_case_list(args.case_list, pilot_only=False)
            failed_cases = [case for case in all_cases if not markdown_exists_for_case(case)]

        if not failed_cases:
            print("No failed cases to retry. All cases currently have markdown output.")
            return

        if args.include_existing:
            cases = failed_cases
        else:
            cases = [case for case in failed_cases if not markdown_exists_for_case(case)]

        if not cases:
            print("All previously failed cases already have markdown output. Use --include-existing to rerun anyway.")
            return
    else:
        run_mode = "pilot" if args.pilot else ("all" if args.all else "range")
        all_cases = load_case_list(args.case_list, pilot_only=args.pilot)
        if args.range:
            start, end = args.range
            cases = all_cases[start - 1:end]
        else:
            cases = all_cases

    metadata_rows = load_metadata(METADATA_FILE)
    metadata_entry_count = len({id(row) for row in metadata_rows.values()})
    print(f"  Metadata entries: {metadata_entry_count}")

    print(f"Cases to process: {len(cases)}")
    print(f"Output directory:  {OUTPUT_DIR.resolve()}")
    print(f"Raw data directory: {RAW_DIR.resolve()}")

    # Process each case
    success = 0
    failed = []
    for i, case_name in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}]", end="")
        if process_case(case_name, metadata_rows):
            success += 1
        else:
            failed.append(case_name)

    should_save_manifest = args.all or args.pilot or bool(args.range) or args.retry_failed
    if should_save_manifest:
        save_failed_manifest(cases, failed, mode=run_mode)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Processed: {len(cases)}")
    print(f"  Success:   {success}")
    print(f"  Failed:    {len(failed)}")
    if failed:
        print(f"\n  Failed cases:")
        for name in failed:
            print(f"    - {name}")
    print(f"\nMarkdown files saved to: {OUTPUT_DIR.resolve()}")
    if should_save_manifest:
        print(f"Failed-case manifest saved to: {FAILED_MANIFEST_FILE.resolve()}")


if __name__ == "__main__":
    main()
