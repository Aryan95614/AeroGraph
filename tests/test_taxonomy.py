"""Tests for taxonomy-backed entity normalization."""

import json
import tempfile
from pathlib import Path

from aerograph.taxonomy import (
    resolve_entity,
    dedup_extractions,
    resolve_all_entities,
    HFACS_TAXONOMY,
    HFACS_LEVEL1,
    PHASE_OF_FLIGHT,
    AIRCRAFT_TYPES,
    WEATHER_CONDITIONS,
    AIRPORT_CODES,
    TAXONOMY_LOOKUPS,
)


class TestResolveAircraft:
    def test_boeing_737_800(self):
        canonical, confidence = resolve_entity("Boeing 737-800", "Aircraft")
        assert canonical == "B738"
        assert confidence > 0.9

    def test_b738(self):
        canonical, confidence = resolve_entity("B738", "Aircraft")
        assert canonical == "B738"
        assert confidence == 1.0

    def test_crj_200(self):
        canonical, confidence = resolve_entity("CRJ-200", "Aircraft")
        assert canonical == "CRJ2"
        assert confidence == 1.0

    def test_embraer_145(self):
        canonical, confidence = resolve_entity("EMB-145", "Aircraft")
        assert canonical == "E145"
        assert confidence == 1.0

    def test_anonymized_aircraft_unchanged(self):
        canonical, confidence = resolve_entity("aircraft x", "Aircraft")
        assert canonical == "aircraft x"
        assert confidence == 0.0

    def test_unknown_aircraft(self):
        canonical, confidence = resolve_entity("xyzzy_unknown_thing", "Aircraft")
        assert canonical == "xyzzy_unknown_thing"
        assert confidence == 0.0


class TestResolveHumanFactors:
    def test_crew_fatigue_maps_to_hfacs(self):
        canonical, confidence = resolve_entity("crew fatigue", "Factor")
        assert canonical == "adverse_physiological_state"
        assert confidence == 1.0

    def test_distraction(self):
        canonical, confidence = resolve_entity("distraction", "Factor")
        assert canonical == "adverse_mental_state"
        assert confidence == 1.0

    def test_communication_breakdown(self):
        canonical, confidence = resolve_entity("communication breakdown", "Factor")
        assert canonical == "crew_resource_management"
        assert confidence == 1.0

    def test_complacency(self):
        canonical, confidence = resolve_entity("complacency", "Factor")
        assert canonical == "adverse_mental_state"
        assert confidence == 1.0

    def test_fatigue_loss_of_sleep(self):
        canonical, confidence = resolve_entity("fatigue - loss of sleep", "Factor")
        assert canonical == "adverse_physiological_state"
        assert confidence == 1.0


class TestResolveAirport:
    def test_atl_to_katl(self):
        canonical, confidence = resolve_entity("ATL", "ATC_Facility")
        assert canonical == "KATL"
        assert confidence == 1.0

    def test_hartsfield(self):
        canonical, confidence = resolve_entity("hartsfield-jackson", "ATC_Facility")
        assert canonical == "KATL"
        assert confidence == 1.0

    def test_sfo(self):
        canonical, confidence = resolve_entity("SFO", "ATC_Facility")
        assert canonical == "KSFO"
        assert confidence == 1.0


class TestResolvePhase:
    def test_approach_phase(self):
        canonical, confidence = resolve_entity("approach phase", "Phase")
        assert canonical == "initial_approach"
        assert confidence == 1.0

    def test_final_approach(self):
        canonical, confidence = resolve_entity("final approach", "Phase")
        assert canonical == "final_approach"
        assert confidence == 1.0

    def test_go_around(self):
        canonical, confidence = resolve_entity("missed approach", "Phase")
        assert canonical == "go_around"
        assert confidence == 1.0


class TestResolveWeather:
    def test_imc_conditions(self):
        canonical, confidence = resolve_entity("IMC conditions", "Weather")
        assert canonical == "IMC"
        assert confidence == 1.0

    def test_thunderstorms(self):
        canonical, confidence = resolve_entity("thunderstorms", "Weather")
        assert canonical == "thunderstorm"
        assert confidence == 1.0

    def test_moderate_turbulence(self):
        canonical, confidence = resolve_entity("moderate turbulence", "Weather")
        assert canonical == "turbulence_moderate"
        assert confidence == 1.0


class TestResolveUnknownType:
    def test_unknown_entity_type(self):
        canonical, confidence = resolve_entity("xyzzy_unknown_thing", "Equipment")
        assert canonical == "xyzzy_unknown_thing"
        assert confidence == 0.0


class TestDedupExtractions:
    def test_removes_exact_duplicates(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            line1 = json.dumps({"report_id": "1", "entities": [], "relations": []})
            line2 = json.dumps({"report_id": "2", "entities": [], "relations": []})
            f.write(line1 + "\n")
            f.write(line2 + "\n")
            f.write(line1 + "\n")  # duplicate
            f.write(line1 + "\n")  # duplicate
            input_path = Path(f.name)

        output_path = input_path.with_suffix(".deduped.jsonl")
        try:
            total, unique, dupes = dedup_extractions(input_path, output_path)
            assert total == 4
            assert unique == 2
            assert dupes == 2
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 2
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def test_no_duplicates(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"report_id": "1", "entities": [], "relations": []}) + "\n")
            f.write(json.dumps({"report_id": "2", "entities": [], "relations": []}) + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_suffix(".deduped.jsonl")
        try:
            total, unique, dupes = dedup_extractions(input_path, output_path)
            assert total == 2
            assert unique == 2
            assert dupes == 0
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class TestResolveAllEntities:
    def test_resolves_known_entities(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            extraction = {
                "report_id": "TEST001",
                "entities": [
                    {"name": "B737-800", "type": "Aircraft", "canonical_name": "b737-800", "report_ids": ["TEST001"]},
                    {"name": "crew fatigue", "type": "Factor", "canonical_name": "crew fatigue", "report_ids": ["TEST001"]},
                    {"name": "ATL", "type": "ATC_Facility", "canonical_name": "atl", "report_ids": ["TEST001"]},
                ],
                "relations": [
                    {"source": "b737-800", "target": "crew fatigue", "type": "INVOLVED", "report_ids": ["TEST001"]},
                ],
            }
            f.write(json.dumps(extraction) + "\n")
            input_path = Path(f.name)

        output_path = input_path.with_suffix(".normalized.jsonl")
        try:
            stats = resolve_all_entities(input_path, output_path, use_embeddings=False)
            assert stats["Aircraft"]["resolved"] >= 1
            assert stats["Factor"]["resolved"] >= 1
            assert stats["ATC_Facility"]["resolved"] >= 1

            # Verify output file
            with open(output_path) as f:
                result = json.loads(f.readline())
            aircraft = [e for e in result["entities"] if e["type"] == "Aircraft"][0]
            assert aircraft["canonical_name"] == "b738"
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


class TestHFACSTaxonomy:
    def test_has_four_level1_categories(self):
        assert len(HFACS_LEVEL1) == 4
        assert "unsafe_acts" in HFACS_LEVEL1
        assert "preconditions_for_unsafe_acts" in HFACS_LEVEL1
        assert "unsafe_supervision" in HFACS_LEVEL1
        assert "organizational_influences" in HFACS_LEVEL1

    def test_all_level2_have_valid_l1(self):
        for canonical, info in HFACS_TAXONOMY.items():
            assert info["l1"] in HFACS_LEVEL1, f"{canonical} has invalid l1: {info['l1']}"

    def test_all_level2_have_aliases(self):
        for canonical, info in HFACS_TAXONOMY.items():
            assert len(info["aliases"]) > 0, f"{canonical} has no aliases"

    def test_level2_count(self):
        # HFACS has 17 Level 2 categories in our taxonomy
        assert len(HFACS_TAXONOMY) >= 15


class TestPhaseOfFlightTaxonomy:
    def test_has_all_icao_phases(self):
        expected = [
            "taxi", "takeoff", "initial_climb", "climb", "cruise",
            "descent", "initial_approach", "final_approach", "landing",
            "go_around", "parking",
        ]
        for phase in expected:
            assert phase in PHASE_OF_FLIGHT, f"Missing ICAO phase: {phase}"

    def test_phases_have_aliases(self):
        for phase, aliases in PHASE_OF_FLIGHT.items():
            assert len(aliases) > 0, f"{phase} has no aliases"


class TestTaxonomyLookups:
    def test_lookups_exist_for_key_types(self):
        assert "Phase" in TAXONOMY_LOOKUPS
        assert "Aircraft" in TAXONOMY_LOOKUPS
        assert "Weather" in TAXONOMY_LOOKUPS
        assert "Factor" in TAXONOMY_LOOKUPS
        assert "ATC_Facility" in TAXONOMY_LOOKUPS

    def test_lookup_returns_canonical(self):
        lookup = TAXONOMY_LOOKUPS["Aircraft"]
        assert lookup["b737"] == "B737"
        assert lookup["crj-200"] == "CRJ2"

    def test_weather_lookup(self):
        lookup = TAXONOMY_LOOKUPS["Weather"]
        assert lookup["thunderstorms"] == "thunderstorm"
        assert lookup["imc"] == "IMC"
