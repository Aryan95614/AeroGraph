"""LLM-based entity and relation extraction from ASRS reports.

Uses Claude claude-sonnet-4-20250514 with structured extraction prompt to pull entities and
relations conforming to an aviation safety ontology.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Aviation Safety Ontology ---

NODE_TYPES = [
    "Aircraft", "Event", "Phase", "Factor", "Component",
    "Outcome", "Recommendation", "ATC_Facility", "Weather",
]

EDGE_TYPES = [
    "CAUSED_BY", "CONTRIBUTED_TO", "OCCURRED_DURING", "INVOLVED",
    "RESOLVED_BY", "PRECEDED_BY", "CO_OCCURRED_WITH",
]


@dataclass
class Entity:
    name: str
    type: str
    canonical_name: str = ""
    report_ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.canonical_name:
            self.canonical_name = self.name.lower().strip()


@dataclass
class Relation:
    source: str
    target: str
    type: str
    report_ids: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    report_id: str
    entities: list[Entity]
    relations: list[Relation]


EXTRACTION_PROMPT = """You are an aviation safety analyst. Extract structured entities and relations from this ASRS incident report.

## Ontology

Entity types: {node_types}
Relation types: {edge_types}

## Rules
- Extract ALL relevant entities mentioned in the report
- Use canonical names (e.g., "B737" not "Boeing 737-800")
- Each relation must connect two extracted entities
- Relations must use one of the defined edge types
- Be thorough but precise — only extract what is explicitly stated or strongly implied

## Report (ACN: {acn})

{text}

## Output Format

Return ONLY valid JSON with this structure:
{{
  "entities": [
    {{"name": "entity name", "type": "EntityType"}}
  ],
  "relations": [
    {{"source": "entity1 name", "target": "entity2 name", "type": "EDGE_TYPE"}}
  ]
}}"""


def get_client():
    """Get Anthropic client."""
    import anthropic
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
def extract_from_report(client, report_id: str, text: str) -> ExtractionResult:
    """Extract entities and relations from a single report using Claude."""
    prompt = EXTRACTION_PROMPT.format(
        node_types=", ".join(NODE_TYPES),
        edge_types=", ".join(EDGE_TYPES),
        acn=report_id,
        text=text,
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text
    parsed = _parse_extraction(raw, report_id)
    return parsed


def _parse_extraction(text: str, report_id: str) -> ExtractionResult:
    """Parse Claude's JSON response into ExtractionResult."""
    # Find JSON in response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return ExtractionResult(report_id=report_id, entities=[], relations=[])

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        cleaned = text[start:end]
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return ExtractionResult(report_id=report_id, entities=[], relations=[])

    entities = []
    for e in data.get("entities", []):
        name = e.get("name", "").strip()
        etype = e.get("type", "Event")
        if name and etype in NODE_TYPES:
            # Normalize canonical name: lowercase, strip, collapse whitespace
            canonical = re.sub(r"\s+", " ", name.lower().strip())
            # Remove trailing punctuation from entity names
            canonical = canonical.rstrip(".,;:!?")
            entities.append(Entity(
                name=name,
                type=etype,
                canonical_name=canonical,
                report_ids=[report_id],
            ))

    relations = []
    entity_names = {e.canonical_name for e in entities}
    for r in data.get("relations", []):
        source = r.get("source", "").strip()
        target = r.get("target", "").strip()
        rtype = r.get("type", "")
        if source and target and rtype in EDGE_TYPES:
            relations.append(Relation(
                source=source.lower().strip(),
                target=target.lower().strip(),
                type=rtype,
                report_ids=[report_id],
            ))

    return ExtractionResult(
        report_id=report_id,
        entities=entities,
        relations=relations,
    )


def _canonicalize_name(name: str) -> str:
    """Apply canonical normalization to an entity name."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = name.rstrip(".,;:!?")
    # Normalize common aviation abbreviations
    replacements = {
        "aircraft": "acft",
        "runway": "rwy",
        "altitude": "alt",
        "frequency": "freq",
        "communication": "comm",
        "maintenance": "maint",
        "controller": "ctlr",
    }
    for full, abbr in replacements.items():
        if name == full:
            name = abbr
    return name


def normalize_entities(entities: list[Entity], threshold: float = 0.85) -> list[Entity]:
    """Deduplicate entities using fuzzy string matching.

    Merges entities of the same type whose canonical names are similar
    above the given threshold (SequenceMatcher ratio). Tuned from 0.90
    down to 0.85 after observing near-duplicate clusters like
    'bird strike' / 'bird strikes' / 'birdstrike' in extraction output.
    """
    if not entities:
        return []

    # Re-canonicalize all names before dedup
    for e in entities:
        e.canonical_name = _canonicalize_name(e.canonical_name)

    # Group by type
    by_type: dict[str, list[Entity]] = {}
    for e in entities:
        by_type.setdefault(e.type, []).append(e)

    merged = []
    for etype, group in by_type.items():
        # Sort by name length (prefer shorter canonical names)
        group.sort(key=lambda e: len(e.canonical_name))
        used = set()

        for i, e1 in enumerate(group):
            if i in used:
                continue
            canonical = e1
            for j, e2 in enumerate(group[i + 1:], start=i + 1):
                if j in used:
                    continue
                ratio = SequenceMatcher(None, e1.canonical_name, e2.canonical_name).ratio()
                if ratio >= threshold:
                    # Merge e2 into canonical
                    canonical.report_ids.extend(e2.report_ids)
                    used.add(j)
            canonical.report_ids = list(set(canonical.report_ids))
            merged.append(canonical)

    return merged


def normalize_relations(
    relations: list[Relation], entity_map: dict[str, str]
) -> list[Relation]:
    """Remap relation endpoints to canonical entity names."""
    normalized = []
    for r in relations:
        source = _resolve_name(r.source, entity_map)
        target = _resolve_name(r.target, entity_map)
        if source and target and source != target:
            normalized.append(Relation(
                source=source,
                target=target,
                type=r.type,
                report_ids=r.report_ids,
            ))
    return normalized


def _resolve_name(name: str, entity_map: dict[str, str]) -> str:
    """Resolve a name to its canonical form using the entity map."""
    canonical = name.lower().strip()
    if canonical in entity_map:
        return entity_map[canonical]
    # Fuzzy match
    best_match = None
    best_score = 0.0
    for key, value in entity_map.items():
        score = SequenceMatcher(None, canonical, key).ratio()
        if score > best_score and score >= 0.80:
            best_score = score
            best_match = value
    return best_match or canonical


def save_extractions(results: list[ExtractionResult], path: Path) -> None:
    """Save extraction results to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for result in results:
            record = {
                "report_id": result.report_id,
                "entities": [asdict(e) for e in result.entities],
                "relations": [asdict(r) for r in result.relations],
            }
            f.write(json.dumps(record) + "\n")


def load_extractions(path: Optional[Path] = None) -> list[ExtractionResult]:
    """Load extraction results from JSONL."""
    if path is None:
        path = PROCESSED_DIR / "extractions.jsonl"
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            entities = [Entity(**e) for e in data["entities"]]
            relations = [Relation(**r) for r in data["relations"]]
            results.append(ExtractionResult(
                report_id=data["report_id"],
                entities=entities,
                relations=relations,
            ))
    return results


def get_processed_acns(path: Path) -> set[str]:
    """Get set of already-processed ACNs for idempotent re-runs."""
    if not path.exists():
        return set()
    acns = set()
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            acns.add(data["report_id"])
    return acns


def run_extraction(
    reports_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    batch_size: int = 20,
) -> list[ExtractionResult]:
    """Run entity extraction on all reports.

    Processes in batches, caches results, skips already-processed ACNs.
    """
    if reports_path is None:
        reports_path = PROCESSED_DIR / "reports.jsonl"
    if output_path is None:
        output_path = PROCESSED_DIR / "extractions.jsonl"

    # Load reports
    from aerograph.ingest import load_reports
    reports = load_reports(reports_path)
    print(f"Loaded {len(reports)} reports for extraction")

    # Check existing extractions for idempotency
    processed_acns = get_processed_acns(output_path)
    remaining = [r for r in reports if r.id not in processed_acns]
    print(f"  {len(processed_acns)} already processed, {len(remaining)} remaining")

    if not remaining:
        return load_extractions(output_path)

    client = get_client()
    all_results = load_extractions(output_path)

    # Process in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_results = []

        for report in batch:
            try:
                result = extract_from_report(client, report.id, report.text)
                batch_results.append(result)
            except Exception as e:
                print(f"  Failed to extract from {report.id}: {e}")
                batch_results.append(ExtractionResult(
                    report_id=report.id, entities=[], relations=[],
                ))

        all_results.extend(batch_results)

        # Append batch results to file
        with open(output_path, "a") as f:
            for result in batch_results:
                record = {
                    "report_id": result.report_id,
                    "entities": [asdict(e) for e in result.entities],
                    "relations": [asdict(r) for r in result.relations],
                }
                f.write(json.dumps(record) + "\n")

        processed_count = len(processed_acns) + batch_start + len(batch)
        print(f"  Extracted {processed_count}/{len(reports)} reports")
        time.sleep(0.5)

    return all_results
