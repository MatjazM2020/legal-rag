"""
ECHR Local Parser — Raw Text to Structured Markdown
====================================================
Converts manually collected ECtHR judgment texts + metadata CSV
into structured markdown files for RAG pipeline.

Usage:
    # Parse all cases that have raw text files + metadata entries
    python parse_cases.py

    # Parse a single case
    python parse_cases.py --case salduz_v_turkey

    # Dry run — show what would be parsed without writing files
    python parse_cases.py --dry-run

    # Verbose mode — show detailed section extraction info
    python parse_cases.py --verbose
"""

import os
import re
import csv
import argparse
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

RAW_DIR = Path("data/raw")         # Where you paste the raw judgment texts
OUTPUT_DIR = Path("data/cases")    # Where structured markdown files go
METADATA_FILE = Path("data/metadata/metadata.csv")

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
# TEXT PARSING — Extract sections from pasted ECtHR judgment text
# =============================================================================

def clean_text(text: str) -> str:
    """Clean up raw pasted text."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive blank lines (keep max 2)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Remove leading/trailing whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def find_section_boundary(text: str, patterns: list[str]) -> int | None:
    """
    Find the position of the first matching pattern in text.
    Patterns are tried in order; returns position of first match.
    """
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.start()
    return None


def extract_sections(text: str) -> dict:
    """
    Parse an ECtHR judgment into structured sections.

    Typical ECtHR judgment structure:
    ─────────────────────────────────
    [Header / Case title]
    PROCEDURE
    THE FACTS
      I. THE CIRCUMSTANCES OF THE CASE
      II. RELEVANT DOMESTIC LAW (AND PRACTICE)
    THE LAW
      I. ALLEGED VIOLATION OF ARTICLE X
      II. ...
    FOR THESE REASONS, THE COURT (UNANIMOUSLY / BY X VOTES TO Y)
      1. Holds that...
      2. Holds that...
    [SEPARATE / DISSENTING / CONCURRING OPINIONS]
    ─────────────────────────────────
    """
    text = clean_text(text)

    # Define section markers — each is a list of regex patterns (tried in order)
    markers = {
        "procedure": [
            r"^PROCEDURE\s*$",
            r"^I\.\s*PROCEDURE",
        ],
        "facts": [
            r"^(?:THE\s+)?FACTS\s*$",
            r"^I\.\s*THE\s+CIRCUMSTANCES\s+OF\s+THE\s+CASE",
            r"^THE\s+CIRCUMSTANCES\s+OF\s+THE\s+CASE",
            r"^AS\s+TO\s+THE\s+FACTS",
        ],
        "relevant_law": [
            r"^(?:II\.\s*)?RELEVANT\s+(?:DOMESTIC\s+)?(?:LAW|LEGAL\s+FRAMEWORK)",
            r"^(?:II\.\s*)?RELEVANT\s+(?:DOMESTIC\s+AND\s+INTERNATIONAL\s+)?LAW\s+AND\s+PRACTICE",
            r"^(?:II\.\s*)?RELEVANT\s+(?:DOMESTIC\s+)?(?:LAW|LEGISLATION)\s+AND\s+PRACTICE",
            r"^(?:B\.\s*)?RELEVANT\s+DOMESTIC\s+LAW",
        ],
        "the_law": [
            r"^THE\s+LAW\s*$",
            r"^AS\s+TO\s+THE\s+LAW",
        ],
        "decision": [
            r"^FOR\s+THESE\s+REASONS,?\s*THE\s+COURT",
            r"^FOR\s+THESE\s+REASONS,?\s*$",
        ],
        "separate_opinions": [
            r"^(?:JOINT\s+)?(?:PARTLY\s+)?(?:CONCURRING|DISSENTING|SEPARATE)\s+OPINION",
            r"^(?:JOINT\s+)?(?:PARTLY\s+)?(?:CONCURRING|DISSENTING|SEPARATE)\s+(?:AND\s+(?:PARTLY\s+)?(?:CONCURRING|DISSENTING)\s+)?OPINION",
        ],
    }

    # Find all section boundaries
    boundaries = []
    for section_name, patterns in markers.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                boundaries.append((match.start(), match.end(), section_name))
                break  # Take first match per pattern group
            else:
                continue
            break  # Found a match for this section, stop trying patterns

    # Sort by position in text
    boundaries.sort(key=lambda x: x[0])

    # Extract text between boundaries
    sections = {
        "procedure": "",
        "facts": "",
        "relevant_law": "",
        "the_law": "",
        "decision": "",
        "separate_opinions": "",
    }

    for i, (start, header_end, section_name) in enumerate(boundaries):
        # Section content starts after the header
        content_start = header_end

        # Section ends at the next boundary (or end of text)
        if i + 1 < len(boundaries):
            content_end = boundaries[i + 1][0]
        else:
            content_end = len(text)

        content = text[content_start:content_end].strip()

        # Append if section already has content (e.g., multiple opinion blocks)
        if sections[section_name]:
            sections[section_name] += "\n\n---\n\n" + content
        else:
            sections[section_name] = content

    # Post-processing: separate "facts" from "relevant_law" if they're combined
    # (some judgments have "THE FACTS" containing both subsections)
    if sections["facts"] and not sections["relevant_law"]:
        # Try to split within the facts section
        relevant_law_split = re.search(
            r"(?:II\.\s*)?RELEVANT\s+(?:DOMESTIC\s+)?(?:LAW|LEGAL\s+FRAMEWORK|LEGISLATION)",
            sections["facts"],
            re.IGNORECASE,
        )
        if relevant_law_split:
            sections["relevant_law"] = sections["facts"][relevant_law_split.start():].strip()
            sections["facts"] = sections["facts"][:relevant_law_split.start()].strip()

    return sections


def extract_articles_from_metadata(articles_field: str) -> list[dict]:
    """
    Parse article references from metadata CSV field.
    Input like: "Art. 6-1; Art. 6-3-c; Art. 13"
    Returns list of dicts with article number and full reference.
    """
    if not articles_field:
        return []

    results = []
    seen = set()
    # Match patterns like "Art. 6", "Art. 6-1", "Art. 6-3-c", "Art. 1 of Protocol 1"
    parts = re.split(r"[;,]", articles_field)
    for part in parts:
        part = part.strip()
        # Extract the main article number
        match = re.search(r"Art\.?\s*(\d{1,2})", part)
        if match:
            art_num = match.group(1)
            if art_num not in seen:
                seen.add(art_num)
                name = ECHR_ARTICLES.get(art_num, "")
                results.append({
                    "number": art_num,
                    "reference": part,
                    "name": name,
                })
    return results


# =============================================================================
# MARKDOWN GENERATION
# =============================================================================

def generate_markdown(metadata: dict, sections: dict) -> str:
    """Generate structured markdown from metadata + parsed sections."""

    case_name = metadata.get("case_name", "Unknown Case")
    app_no = metadata.get("application_number", "N/A")
    judgment_date = metadata.get("judgment_date", "N/A")
    respondent = metadata.get("respondent_state", "N/A")
    chamber = metadata.get("chamber", "N/A")
    conclusion = metadata.get("conclusion", "N/A")
    importance = metadata.get("importance", "N/A")
    ecli = metadata.get("ecli", "N/A") or "N/A"
    sep_opinion = metadata.get("separate_opinion", "No")
    keywords = metadata.get("keywords", "N/A") or "N/A"

    # Parse articles
    articles = extract_articles_from_metadata(metadata.get("articles", ""))
    articles_str = ", ".join(
        f"Art. {a['number']} ({a['name']})" if a["name"] else f"Art. {a['number']}"
        for a in articles
    ) or "N/A"

    # Section content with fallbacks
    facts = sections.get("facts", "").strip()
    relevant_law = sections.get("relevant_law", "").strip()
    the_law = sections.get("the_law", "").strip()
    decision = sections.get("decision", "").strip()
    separate_opinions = sections.get("separate_opinions", "").strip()

    # Build markdown
    md = f"""# {case_name}

## Metadata
- **Application Number:** {app_no}
- **ECLI:** {ecli}
- **Court:** European Court of Human Rights
- **Chamber:** {chamber}
- **Date of Judgment:** {judgment_date}
- **Respondent State:** {respondent}
- **Importance:** {importance}
- **Articles Cited:** {articles_str}
- **Conclusion:** {conclusion}
- **Separate Opinion:** {sep_opinion}
- **Keywords:** {keywords}

## Facts

{facts if facts else '_Section not automatically extracted. Check the raw text file in data/raw/._'}

## Relevant Legal Framework

{relevant_law if relevant_law else '_No separate legal framework section identified in the judgment text._'}

## Court\'s Reasoning

{the_law if the_law else '_Reasoning section not automatically extracted. Check the raw text file in data/raw/._'}

## Decision

{decision if decision else '_Decision section not automatically extracted. Search for "FOR THESE REASONS" in the raw text file._'}
"""

    if separate_opinions:
        md += f"""
## Separate Opinions

{separate_opinions}
"""

    md += """
## Significance

_TODO: Add a brief note on why this case is a landmark/important precedent and what doctrinal principle it established._

---
*Generated from manually collected HUDOC data. Review all sections for accuracy.*
"""

    return md


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def load_metadata(filepath: Path) -> dict:
    """Load metadata CSV into a dict keyed by file_id."""
    metadata = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_id = row["file_id"].strip()
            metadata[file_id] = row
    return metadata


def get_available_cases(raw_dir: Path, metadata: dict) -> list[tuple[str, Path]]:
    """Find cases that have both a raw text file and a metadata entry."""
    available = []
    for txt_file in sorted(raw_dir.glob("*.txt")):
        file_id = txt_file.stem  # filename without extension
        if file_id in metadata:
            available.append((file_id, txt_file))
        else:
            print(f"  [WARN] Raw file '{txt_file.name}' has no metadata entry — skipping")
    return available


def process_case(file_id: str, raw_path: Path, metadata: dict, verbose: bool = False) -> bool:
    """Process a single case: parse raw text + metadata → markdown."""

    case_meta = metadata.get(file_id)
    if not case_meta:
        print(f"  [ERROR] No metadata found for '{file_id}'")
        return False

    case_name = case_meta.get("case_name", file_id)
    print(f"\n  Processing: {case_name}")

    # Read raw text
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except UnicodeDecodeError:
        # Try with latin-1 as fallback
        with open(raw_path, "r", encoding="latin-1") as f:
            raw_text = f.read()

    if len(raw_text.strip()) < 100:
        print(f"  [WARN] Raw text file seems too short ({len(raw_text)} chars) — check the file")

    # Parse sections
    sections = extract_sections(raw_text)

    # Report extraction results
    section_names = ["procedure", "facts", "relevant_law", "the_law", "decision", "separate_opinions"]
    for name in section_names:
        content = sections.get(name, "")
        if verbose or not content:
            status = "✓" if content else "✗ MISSING"
            length = len(content) if content else 0
            print(f"    {status:12s} {name}: {length} chars")

    # Check critical sections
    missing_critical = []
    if not sections["facts"]:
        missing_critical.append("facts")
    if not sections["the_law"]:
        missing_critical.append("the_law (reasoning)")
    if not sections["decision"]:
        missing_critical.append("decision")

    if missing_critical:
        print(f"  [WARN] Missing critical sections: {', '.join(missing_critical)}")
        print(f"         The markdown will have placeholders. Review and fix manually.")

    # Generate markdown
    md_content = generate_markdown(case_meta, sections)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{file_id}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"  ✓ Saved: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Parse manually collected ECtHR judgments into structured markdown"
    )
    parser.add_argument("--case", type=str, help="Parse a single case by file_id")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be parsed")
    parser.add_argument("--verbose", action="store_true", help="Show detailed extraction info")

    args = parser.parse_args()

    print("=" * 60)
    print("ECHR Local Parser — Raw Text → Structured Markdown")
    print("=" * 60)

    # Check directories exist
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True)
        print(f"\n  Created {RAW_DIR}/ — paste your raw text files here")
        print(f"  See COLLECTION_GUIDE.md for instructions")
        return

    if not METADATA_FILE.exists():
        print(f"\n  [ERROR] metadata.csv not found!")
        print(f"  Copy the template and fill in case details.")
        return

    # Load metadata
    metadata = load_metadata(METADATA_FILE)
    print(f"  Metadata entries: {len(metadata)}")

    # Find available cases
    if args.case:
        raw_path = RAW_DIR / f"{args.case}.txt"
        if not raw_path.exists():
            print(f"\n  [ERROR] Raw file not found: {raw_path}")
            print(f"  Make sure you saved the text as: {raw_path}")
            return
        cases = [(args.case, raw_path)]
    else:
        cases = get_available_cases(RAW_DIR, metadata)

    if not cases:
        print(f"\n  No cases ready to parse!")
        print(f"  1. Collect raw text from HUDOC (see COLLECTION_GUIDE.md)")
        print(f"  2. Save to data/raw/<file_id>.txt")
        print(f"  3. Add metadata row to metadata.csv")
        print(f"  4. Run this script again")
        return

    print(f"  Cases to parse: {len(cases)}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Would parse these cases:")
        for file_id, path in cases:
            case_name = metadata.get(file_id, {}).get("case_name", file_id)
            print(f"    - {case_name} ({path})")
        return

    # Process cases
    success = 0
    failed = []
    for file_id, raw_path in cases:
        if process_case(file_id, raw_path, metadata, verbose=args.verbose):
            success += 1
        else:
            failed.append(file_id)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE — {success}/{len(cases)} cases parsed successfully")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
