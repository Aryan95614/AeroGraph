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

DATA_DIR = Path(os.environ.get("AEROGRAPH_DATA_DIR", Path(__file__).parent.parent.parent / "data"))
PROCESSED_DIR = DATA_DIR / "processed"

# --- Aviation Safety Ontology ---

NODE_TYPES = [
    "Aircraft", "Event", "Phase", "Factor", "Component",
    "Outcome", "Recommendation", "ATC_Facility", "Weather",
    "TimePeriod",
]

EDGE_TYPES = [
    "CAUSED_BY", "CONTRIBUTED_TO", "OCCURRED_DURING", "INVOLVED",
    "RESOLVED_BY", "PRECEDED_BY", "CO_OCCURRED_WITH",
    "TEMPORAL_SEQUENCE",
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
- Extract temporal sequences: when the narrative describes events in order (e.g., "X happened before Y", "after the go-around, Z occurred", "following the missed approach, the crew did W"), use TEMPORAL_SEQUENCE edges from the earlier event to the later event
- Use the TimePeriod entity type for explicit time references (e.g., "during descent", "at 1430Z", "night operations")

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
        if (source and target and rtype in EDGE_TYPES
                and source.lower().strip() in entity_names
                and target.lower().strip() in entity_names):
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


def _differ_only_by_number(name1: str, name2: str) -> bool:
    """Check if two entity names differ only by a numeric or alphanumeric identifier.

    Prevents merging 'engine 1 failure' with 'engine 2 failure',
    'runway 28L' with 'runway 28R', etc.
    """
    # Match numbers optionally followed by a letter (e.g., "28L", "28R")
    tokens1 = re.findall(r"\d+[a-zA-Z]?", name1)
    tokens2 = re.findall(r"\d+[a-zA-Z]?", name2)
    if not tokens1 and not tokens2:
        return False
    stripped1 = re.sub(r"\d+[a-zA-Z]?", "#", name1)
    stripped2 = re.sub(r"\d+[a-zA-Z]?", "#", name2)
    return stripped1 == stripped2 and tokens1 != tokens2


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
                # Don't merge entities that differ only by a number
                # e.g., "engine 1 failure" vs "engine 2 failure"
                if _differ_only_by_number(e1.canonical_name, e2.canonical_name):
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


def _extract_one(args):
    """Extract from a single report — used by ThreadPoolExecutor."""
    client, report_id, text = args
    try:
        result = extract_from_report(client, report_id, text)
        return result
    except Exception as e:
        print(f"  Failed to extract from {report_id}: {e}", flush=True)
        return ExtractionResult(report_id=report_id, entities=[], relations=[])


def run_extraction(
    reports_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    batch_size: int = 20,
    max_workers: int = 5,
    max_reports: int | None = None,
) -> list[ExtractionResult]:
    """Run entity extraction on all reports.

    Processes in batches with concurrent API calls for speed.
    Caches results, skips already-processed ACNs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if reports_path is None:
        reports_path = PROCESSED_DIR / "reports.jsonl"
    if output_path is None:
        output_path = PROCESSED_DIR / "extractions.jsonl"

    # Load reports
    from aerograph.ingest import load_reports
    reports = load_reports(reports_path)
    print(f"Loaded {len(reports)} reports for extraction", flush=True)

    # Check existing extractions for idempotency
    processed_acns = get_processed_acns(output_path)
    remaining = [r for r in reports if r.id not in processed_acns]
    if max_reports is not None:
        remaining = remaining[:max_reports]
    print(f"  {len(processed_acns)} already processed, {len(remaining)} remaining", flush=True)

    if not remaining:
        return load_extractions(output_path)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Cannot run extraction. "
            "Set it in .env or as an environment variable."
        )

    client = get_client()
    all_results = load_extractions(output_path)
    total = len(reports)
    done = len(processed_acns)
    t0 = time.time()

    # Process in batches with concurrent workers
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_results = []

        args_list = [(client, r.id, r.text) for r in batch]
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_extract_one, a): a[1] for a in args_list}
            for future in as_completed(futures):
                batch_results.append(future.result())

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
            f.flush()

        done += len(batch)
        elapsed = time.time() - t0
        rate = done - len(processed_acns)
        if rate > 0:
            per_report = elapsed / rate
            eta_min = (total - done) * per_report / 60
            print(f"  [{done}/{total}] {len(batch)} extracted | "
                  f"{per_report:.1f}s/report | ETA {eta_min:.0f}min", flush=True)
        else:
            print(f"  [{done}/{total}] {len(batch)} extracted", flush=True)

    elapsed = time.time() - t0
    total_entities = sum(len(r.entities) for r in all_results)
    total_relations = sum(len(r.relations) for r in all_results)
    print(f"\nExtraction complete: {len(all_results)} reports, "
          f"{total_entities} entities, {total_relations} relations "
          f"in {elapsed/60:.1f}min", flush=True)

    return all_results
