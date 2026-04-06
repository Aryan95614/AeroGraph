"""Tests for entity extraction."""

import json

from aerograph.extract import (
    Entity,
    Relation,
    ExtractionResult,
    _parse_extraction,
    _differ_only_by_number,
    normalize_entities,
    normalize_relations,
    _resolve_name,
    NODE_TYPES,
    EDGE_TYPES,
)


class TestOntology:
    def test_node_types(self):
        assert "Aircraft" in NODE_TYPES
        assert "Event" in NODE_TYPES
        assert "Factor" in NODE_TYPES
        assert len(NODE_TYPES) == 10

    def test_edge_types(self):
        assert "CAUSED_BY" in EDGE_TYPES
        assert "CONTRIBUTED_TO" in EDGE_TYPES
        assert len(EDGE_TYPES) == 8


class TestParseExtraction:
    def test_valid_json(self):
        text = json.dumps({
            "entities": [
                {"name": "B737", "type": "Aircraft"},
                {"name": "Bird Strike", "type": "Event"},
            ],
            "relations": [
                {"source": "Bird Strike", "target": "B737", "type": "INVOLVED"},
            ],
        })
        result = _parse_extraction(text, "ACN001")
        assert len(result.entities) == 2
        assert len(result.relations) == 1
        assert result.entities[0].canonical_name == "b737"

    def test_json_with_surrounding_text(self):
        text = 'Here is the extraction:\n{"entities": [{"name": "Engine", "type": "Component"}], "relations": []}\nDone.'
        result = _parse_extraction(text, "ACN002")
        assert len(result.entities) == 1

    def test_invalid_entity_type_filtered(self):
        text = json.dumps({
            "entities": [{"name": "Foo", "type": "InvalidType"}],
            "relations": [],
        })
        result = _parse_extraction(text, "ACN003")
        assert len(result.entities) == 0

    def test_invalid_edge_type_filtered(self):
        text = json.dumps({
            "entities": [
                {"name": "A", "type": "Event"},
                {"name": "B", "type": "Factor"},
            ],
            "relations": [
                {"source": "A", "target": "B", "type": "INVALID_EDGE"},
            ],
        })
        result = _parse_extraction(text, "ACN004")
        assert len(result.relations) == 0

    def test_empty_response(self):
        result = _parse_extraction("No JSON here", "ACN005")
        assert len(result.entities) == 0
        assert len(result.relations) == 0


class TestNormalizeEntities:
    def test_dedup_similar_names(self):
        entities = [
            Entity(name="bird strike", type="Event", report_ids=["1"]),
            Entity(name="bird strikes", type="Event", report_ids=["2"]),
        ]
        merged = normalize_entities(entities, threshold=0.85)
        assert len(merged) == 1
        assert set(merged[0].report_ids) == {"1", "2"}

    def test_different_types_not_merged(self):
        entities = [
            Entity(name="engine", type="Component", report_ids=["1"]),
            Entity(name="engine", type="Event", report_ids=["2"]),
        ]
        merged = normalize_entities(entities, threshold=0.85)
        assert len(merged) == 2

    def test_numbered_entities_not_merged(self):
        entities = [
            Entity(name="engine 1 failure", type="Event", report_ids=["1"]),
            Entity(name="engine 2 failure", type="Event", report_ids=["2"]),
        ]
        merged = normalize_entities(entities, threshold=0.85)
        assert len(merged) == 2

    def test_runway_numbers_not_merged(self):
        entities = [
            Entity(name="runway 28L", type="Component", report_ids=["1"]),
            Entity(name="runway 28R", type="Component", report_ids=["2"]),
        ]
        merged = normalize_entities(entities, threshold=0.85)
        assert len(merged) == 2

    def test_dissimilar_names_not_merged(self):
        entities = [
            Entity(name="bird strike", type="Event", report_ids=["1"]),
            Entity(name="engine failure", type="Event", report_ids=["2"]),
        ]
        merged = normalize_entities(entities, threshold=0.85)
        assert len(merged) == 2

    def test_empty_input(self):
        assert normalize_entities([]) == []


class TestNormalizeRelations:
    def test_remaps_endpoints(self):
        entity_map = {
            "bird strike": "bird strike",
            "bird strikes": "bird strike",
            "b737": "b737",
        }
        relations = [
            Relation(source="bird strikes", target="b737", type="INVOLVED", report_ids=["1"]),
        ]
        normalized = normalize_relations(relations, entity_map)
        assert len(normalized) == 1
        assert normalized[0].source == "bird strike"

    def test_self_loop_removed(self):
        entity_map = {"engine": "engine", "engines": "engine"}
        relations = [
            Relation(source="engine", target="engines", type="CAUSED_BY", report_ids=["1"]),
        ]
        normalized = normalize_relations(relations, entity_map)
        assert len(normalized) == 0


class TestDifferOnlyByNumber:
    def test_engine_numbers(self):
        assert _differ_only_by_number("engine 1 failure", "engine 2 failure") is True

    def test_runway_designators(self):
        assert _differ_only_by_number("runway 28l", "runway 28r") is True  # different designators
        assert _differ_only_by_number("runway 28", "runway 10") is True

    def test_no_numbers(self):
        assert _differ_only_by_number("bird strike", "bird strikes") is False

    def test_same_numbers(self):
        assert _differ_only_by_number("engine 1", "engine 1") is False


class TestResolveName:
    def test_exact_match(self):
        entity_map = {"bird strike": "bird strike", "b737": "b737"}
        assert _resolve_name("bird strike", entity_map) == "bird strike"

    def test_fuzzy_match(self):
        entity_map = {"bird strike": "bird strike"}
        result = _resolve_name("bird strikes", entity_map)
        assert result == "bird strike"
