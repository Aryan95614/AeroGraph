"""Tests for the ASRS ingestion pipeline."""

import json
import tempfile
from pathlib import Path

from aerograph.ingest import (
    normalize_aircraft,
    extract_phase,
    extract_anomaly,
    clean_narrative,
    process_reports,
    save_reports,
    load_reports,
    ASRSReport,
    SyntheticGenerator,
)


class TestNormalizeAircraft:
    def test_boeing_variants(self):
        assert normalize_aircraft("Boeing 737") == "B737"
        assert normalize_aircraft("737-800") == "B737"
        assert normalize_aircraft("b737-800") == "B737"
        assert normalize_aircraft("737") == "B737"

    def test_airbus_variants(self):
        assert normalize_aircraft("Airbus A320") == "A320"
        assert normalize_aircraft("a320") == "A320"
        assert normalize_aircraft("A320NEO") == "A320NEO"

    def test_ga_aircraft(self):
        assert normalize_aircraft("Cessna 172") == "C172"
        assert normalize_aircraft("skyhawk") == "C172"
        assert normalize_aircraft("PA-28") == "PA28"

    def test_unknown_passthrough(self):
        result = normalize_aircraft("XYZ-999")
        assert result == "XYZ-999"


class TestExtractPhase:
    def test_finds_phase(self):
        assert extract_phase("During the approach phase, the aircraft...") == "approach"
        assert extract_phase("While in cruise at FL350...") == "cruise"
        assert extract_phase("On takeoff roll from runway 28L...") == "takeoff"

    def test_unknown_phase(self):
        assert extract_phase("Something happened.") == "unknown"


class TestExtractAnomaly:
    def test_finds_anomaly(self):
        assert extract_anomaly("A bird strike occurred...") == "bird strike"
        assert extract_anomaly("The engine failure light illuminated...") == "engine failure"

    def test_unknown_anomaly(self):
        assert extract_anomaly("Normal operations.") == "other"


class TestCleanNarrative:
    def test_removes_boilerplate(self):
        text = "ASRS Report #12345 --- Some actual content here."
        cleaned = clean_narrative(text)
        assert "ASRS Report" not in cleaned
        assert "actual content" in cleaned
        assert "---" not in cleaned

    def test_normalizes_whitespace(self):
        text = "Multiple   spaces   and\nnewlines\n\nhere."
        cleaned = clean_narrative(text)
        assert "  " not in cleaned


class TestProcessReports:
    def test_basic_processing(self):
        reports = [
            ASRSReport(
                acn="1850001",
                date="2024-06-15",
                aircraft_type="B737",
                phase_of_flight="approach",
                anomaly_type="bird strike",
                contributing_factors=["weather", "fatigue"],
                narrative="During approach to runway 28L at KLAX, we encountered a large bird strike at approximately 3000 feet AGL. The First Officer handled communications while I flew the approach.",
                synopsis="Bird strike during approach at KLAX.",
                is_synthetic=True,
            )
        ]
        processed = process_reports(reports)
        assert len(processed) == 1
        assert processed[0].id == "1850001"
        assert processed[0].metadata["aircraft_type"] == "B737"
        assert processed[0].metadata["is_synthetic"] is True

    def test_filters_short_narratives(self):
        reports = [
            ASRSReport(
                acn="1850002", date="2024-01-01",
                aircraft_type="B737", phase_of_flight="cruise",
                anomaly_type="other", contributing_factors=[],
                narrative="Too short.",
            )
        ]
        processed = process_reports(reports)
        assert len(processed) == 0


class TestSaveLoadRoundtrip:
    def test_roundtrip(self, tmp_path):
        from aerograph.ingest import ProcessedReport
        reports = [
            ProcessedReport(
                id="1850001",
                text="Test narrative for roundtrip testing.",
                metadata={"acn": "1850001", "aircraft_type": "B737"},
            )
        ]
        path = tmp_path / "test_reports.jsonl"
        save_reports(reports, path)
        loaded = load_reports(path)
        assert len(loaded) == 1
        assert loaded[0].id == "1850001"
        assert loaded[0].metadata["aircraft_type"] == "B737"


class TestSyntheticGenerator:
    def test_template_generation(self):
        gen = SyntheticGenerator()
        reports = gen._generate_template_based(10)
        assert len(reports) == 10
        for r in reports:
            assert r.is_synthetic
            assert len(r.narrative) > 50
            assert r.acn.startswith("185")
