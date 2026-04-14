# ECHR Legal Data Pipeline (Local / Manual)

Collects ECtHR case law from HUDOC manually and converts it to structured markdown for the RAG pipeline.

## Directory Structure

```
├── parse_cases.py          # Parser: raw text → structured markdown
├── metadata.csv            # Case metadata (you fill this in)
├── case_list.txt           # All 50 target cases
├── COLLECTION_GUIDE.md     # Step-by-step guide for collecting data from HUDOC
├── requirements.txt        # No external deps needed (stdlib only)
├── README.md               # This file
└── data/
    ├── raw/                # Paste raw judgment texts here (.txt files)
    ├── cases/              # Output: structured markdown files
    └── metadata/           # ECHR articles reference
```

## Workflow

### Step 1: Collect raw data from HUDOC

Follow `COLLECTION_GUIDE.md`. For each case:
1. Search on https://hudoc.echr.coe.int/
2. Copy the full judgment text → save as `data/raw/<file_id>.txt`
3. Copy metadata from the case detail panel → add a row to `metadata.csv`

### Step 2: Run the parser

```bash
# Parse all cases that have both a raw file + metadata entry
python parse_cases.py

# Parse just one case
python parse_cases.py --case salduz_v_turkey

# See detailed extraction info
python parse_cases.py --verbose

# Preview what would be parsed (no files written)
python parse_cases.py --dry-run
```

### Step 3: Review the output

Check `data/cases/` — each markdown file should have:
- ✅ Metadata block with all case details
- ✅ Facts section
- ✅ Relevant legal framework
- ✅ Court's reasoning (THE LAW section)
- ✅ Decision (FOR THESE REASONS section)
- ⬜ Significance (you add this manually)

### Step 4: Push to git

```bash
git add data/cases/ data/raw/ metadata.csv
git commit -m "Add pilot cases (5 ECtHR judgments)"
git push
```

## Pilot Cases (start here)

| # | Case | file_id | Articles |
|---|------|---------|----------|
| 1 | Salduz v. Turkey | `salduz_v_turkey` | Art. 6 |
| 2 | Chahal v. the United Kingdom | `chahal_v_the_united_kingdom` | Art. 3, 5, 13 |
| 3 | Loizidou v. Turkey | `loizidou_v_turkey` | Art. 1, P1-1 |
| 4 | Nachova and Others v. Bulgaria | `nachova_and_others_v_bulgaria` | Art. 2, 14 |
| 5 | Lukenda v. Slovenia | `lukenda_v_slovenia` | Art. 6, 13 |

## How the markdown maps to the justification engine

```
Facts section           → FACTS        (what happened)
Relevant Legal Framework → NORMS        (which laws apply)
Court's Reasoning       → INTERPRETATION (how the court analyzed it)
Decision                → CONCLUSION    (what the court decided)
```

This is the exact chain your justification engine needs:
**facts → norms → interpretation → conclusion**
