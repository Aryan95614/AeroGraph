"""ASRS report ingestion pipeline.

Supports three data sources in priority order:
  A) Direct ASRS database download (ACN range 1800000-1900000)
  B) ASRS Callback newsletter PDF parsing
  C) Synthetic report generation via Claude (clearly labeled)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

AIRCRAFT_NORMALIZATIONS = {
    "boeing 737": "B737", "737": "B737", "737-800": "B737", "b737-800": "B737",
    "737-700": "B737", "b737-700": "B737", "737-900": "B737", "b737-900": "B737",
    "737-max": "B737MAX", "737 max": "B737MAX", "b737max": "B737MAX",
    "boeing 747": "B747", "747": "B747", "747-400": "B747", "b747-400": "B747",
    "boeing 757": "B757", "757": "B757", "757-200": "B757",
    "boeing 767": "B767", "767": "B767", "767-300": "B767",
    "boeing 777": "B777", "777": "B777", "777-200": "B777", "777-300": "B777",
    "boeing 787": "B787", "787": "B787", "787-9": "B787", "dreamliner": "B787",
    "airbus a320": "A320", "a320": "A320", "a320neo": "A320NEO",
    "airbus a319": "A319", "a319": "A319",
    "airbus a321": "A321", "a321": "A321", "a321neo": "A321NEO",
    "airbus a330": "A330", "a330": "A330",
    "airbus a340": "A340", "a340": "A340",
    "airbus a350": "A350", "a350": "A350",
    "airbus a380": "A380", "a380": "A380",
    "cessna 172": "C172", "c172": "C172", "skyhawk": "C172",
    "cessna 182": "C182", "c182": "C182",
    "cessna 210": "C210", "c210": "C210",
    "piper pa-28": "PA28", "pa-28": "PA28", "cherokee": "PA28",
    "beechcraft king air": "BE-KA", "king air": "BE-KA",
    "embraer 175": "E175", "e175": "E175", "erj-175": "E175",
    "embraer 190": "E190", "e190": "E190",
    "crj-200": "CRJ200", "crj200": "CRJ200",
    "crj-700": "CRJ700", "crj700": "CRJ700",
    "crj-900": "CRJ900", "crj900": "CRJ900",
    "md-80": "MD80", "md80": "MD80", "dc-9": "MD80",
    "dash 8": "DHC8", "dhc-8": "DHC8", "q400": "DHC8",
}

PHASES_OF_FLIGHT = [
    "taxi", "takeoff", "initial climb", "climb", "cruise",
    "descent", "approach", "final approach", "landing", "go-around",
    "parked", "pushback", "holding",
]

ANOMALY_TYPES = [
    "runway incursion", "runway excursion", "bird strike",
    "engine failure", "hydraulic failure", "electrical failure",
    "pressurization loss", "turbulence encounter", "wind shear",
    "tcas ra", "tcas ta", "altitude deviation", "heading deviation",
    "speed deviation", "communication failure", "navigation error",
    "fuel issue", "icing", "lightning strike", "smoke/fire",
    "gear malfunction", "flap malfunction", "autopilot disconnect",
    "go-around", "missed approach", "atc error", "pilot deviation",
    "maintenance error", "fatigue", "controlled flight into terrain",
    "near midair collision", "ground conflict",
]


@dataclass
class ASRSReport:
    acn: str
    date: str
    aircraft_type: str
    phase_of_flight: str
    anomaly_type: str
    contributing_factors: list[str]
    narrative: str
    synopsis: str = ""
    is_synthetic: bool = False


@dataclass
class ProcessedReport:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def normalize_aircraft(raw: str) -> str:
    """Normalize aircraft type string to canonical form."""
    key = raw.lower().strip()
    if key in AIRCRAFT_NORMALIZATIONS:
        return AIRCRAFT_NORMALIZATIONS[key]
    # Try partial match
    for pattern, canonical in AIRCRAFT_NORMALIZATIONS.items():
        if pattern in key or key in pattern:
            return canonical
    return raw.upper().strip()


def extract_phase(text: str) -> str:
    """Extract phase of flight from narrative text."""
    lower = text.lower()
    for phase in PHASES_OF_FLIGHT:
        if phase in lower:
            return phase
    return "unknown"


def extract_anomaly(text: str) -> str:
    """Extract anomaly type from narrative text."""
    lower = text.lower()
    for anomaly in ANOMALY_TYPES:
        if anomaly in lower:
            return anomaly
    return "other"


def clean_narrative(text: str) -> str:
    """Strip boilerplate headers/footers and normalize whitespace."""
    # Remove common ASRS boilerplate patterns
    text = re.sub(r"ASRS\s+Report\s+#?\d*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Callback\s+Conversation", "", text, flags=re.IGNORECASE)
    text = re.sub(r"NASA\s+Aviation\s+Safety\s+Reporting\s+System", "", text, flags=re.IGNORECASE)
    text = re.sub(r"-{3,}", "", text)
    text = re.sub(r"={3,}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ASRSDownloader:
    """Attempt to download ASRS reports from NASA database."""

    BASE_URL = "https://asrs.arc.nasa.gov"
    SEARCH_URL = f"{BASE_URL}/search/database.html"

    def __init__(self, acn_start: int = 1800000, acn_end: int = 1900000):
        self.acn_start = acn_start
        self.acn_end = acn_end

    def download(self, target_count: int = 5000) -> list[dict]:
        """Try to download reports. Returns empty list on failure."""
        try:
            reports = []
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                # Try the ASRS search endpoint
                resp = client.get(self.SEARCH_URL)
                if resp.status_code != 200:
                    print(f"ASRS search page returned {resp.status_code}, falling back")
                    return []

                # ASRS uses a form-based search; attempt ACN-based queries
                batch_size = 100
                for start in range(self.acn_start, self.acn_end, batch_size):
                    end = min(start + batch_size, self.acn_end)
                    try:
                        resp = client.post(
                            f"{self.BASE_URL}/search/dbretrieve.html",
                            data={
                                "acn_start": str(start),
                                "acn_end": str(end),
                                "output_format": "json",
                            },
                            timeout=60.0,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            if isinstance(data, list):
                                reports.extend(data)
                            elif isinstance(data, dict) and "reports" in data:
                                reports.extend(data["reports"])
                    except (httpx.HTTPError, json.JSONDecodeError):
                        continue

                    if len(reports) >= target_count:
                        break
                    time.sleep(0.5)  # Rate limiting

            return reports[:target_count]
        except Exception as e:
            print(f"ASRS download failed: {e}")
            return []


class SyntheticGenerator:
    """Generate realistic synthetic ASRS reports using Claude."""

    AIRCRAFT_TYPES = ["B737", "A320", "B777", "E175", "CRJ900", "C172", "PA28", "B787", "A321", "DHC8"]
    WEATHER_CONDITIONS = ["VMC", "IMC", "marginal VFR", "thunderstorms", "icing", "clear", "fog", "low visibility"]
    AIRPORTS = ["KLAX", "KJFK", "KORD", "KATL", "KDEN", "KSFO", "KDFW", "KMIA", "KBOS", "KSEA",
                "KPHX", "KEWR", "KMSP", "KDTW", "KPHL", "KSLC", "KSAN", "KTPA", "KLAS", "KMCO"]

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    def generate_batch(self, count: int = 500) -> list[ASRSReport]:
        """Generate synthetic reports using Claude API."""
        if not self.api_key:
            print("No ANTHROPIC_API_KEY set, using template-based synthetic generation")
            return self._generate_template_based(count)

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            reports = []
            batch_size = 10

            for batch_start in range(0, count, batch_size):
                current_batch = min(batch_size, count - batch_start)
                prompt = self._build_generation_prompt(current_batch, batch_start)

                try:
                    response = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=8000,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    batch_reports = self._parse_generated_reports(
                        response.content[0].text, batch_start
                    )
                    reports.extend(batch_reports)
                    print(f"  Generated {len(reports)}/{count} synthetic reports")
                except Exception as e:
                    print(f"  Claude generation failed for batch {batch_start}: {e}")
                    remaining = count - len(reports)
                    reports.extend(self._generate_template_based(remaining))
                    break

                time.sleep(1.0)  # Rate limit

            return reports[:count]
        except ImportError:
            print("anthropic package not installed, using template-based generation")
            return self._generate_template_based(count)

    def _build_generation_prompt(self, count: int, offset: int) -> str:
        return f"""Generate {count} realistic NASA ASRS (Aviation Safety Reporting System) incident reports.
Each report should be a JSON object in a JSON array. Start ACN numbering at {1850000 + offset}.

Each report must have these fields:
- acn: string (e.g., "1850001")
- date: string (YYYY-MM-DD format, dates in 2024-2025)
- aircraft_type: one of {self.AIRCRAFT_TYPES}
- phase_of_flight: one of {PHASES_OF_FLIGHT}
- anomaly_type: one of {ANOMALY_TYPES}
- contributing_factors: list of 2-4 strings describing contributing factors
- narrative: realistic first-person pilot/controller narrative (150-400 words)
- synopsis: one-sentence summary

Make each report unique with different scenarios. Include realistic aviation terminology,
ATC callsigns, altitude/speed values, and weather conditions. The narratives should read
like real safety reports — factual, detailed, focused on what happened and why.

Return ONLY a valid JSON array, no other text."""

    def _parse_generated_reports(self, text: str, offset: int) -> list[ASRSReport]:
        """Parse Claude's response into ASRSReport objects."""
        # Find JSON array in response
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        try:
            data = json.loads(text[start:end])
            reports = []
            for item in data:
                report = ASRSReport(
                    acn=str(item.get("acn", f"SYN{1850000 + offset + len(reports)}")),
                    date=item.get("date", "2024-06-15"),
                    aircraft_type=normalize_aircraft(item.get("aircraft_type", "B737")),
                    phase_of_flight=item.get("phase_of_flight", "cruise"),
                    anomaly_type=item.get("anomaly_type", "other"),
                    contributing_factors=item.get("contributing_factors", []),
                    narrative=item.get("narrative", ""),
                    synopsis=item.get("synopsis", ""),
                    is_synthetic=True,
                )
                reports.append(report)
            return reports
        except json.JSONDecodeError:
            return []

    def _generate_template_based(self, count: int) -> list[ASRSReport]:
        """Fallback: generate reports from templates without LLM."""
        import random
        random.seed(42)

        templates = self._load_templates()
        reports = []

        for i in range(count):
            t = random.choice(templates)
            acn = str(1850000 + i)
            aircraft = random.choice(self.AIRCRAFT_TYPES)
            phase = random.choice(PHASES_OF_FLIGHT)
            anomaly = random.choice(ANOMALY_TYPES)
            airport = random.choice(self.AIRPORTS)
            altitude = random.choice(["3000", "5000", "10000", "18000", "25000", "35000", "FL350", "FL370", "FL410"])
            weather = random.choice(self.WEATHER_CONDITIONS)

            narrative = t.format(
                aircraft=aircraft, phase=phase, anomaly=anomaly,
                airport=airport, altitude=altitude, weather=weather,
                callsign=f"N{random.randint(100,999)}AA",
            )

            factors = random.sample([
                "human factors", "fatigue", "communication breakdown",
                "equipment malfunction", "weather", "ATC workload",
                "inadequate training", "procedural deviation",
                "situational awareness loss", "time pressure",
                "maintenance oversight", "crew resource management",
            ], k=random.randint(2, 4))

            month = random.randint(1, 12)
            day = random.randint(1, 28)
            year = random.choice([2024, 2025])

            reports.append(ASRSReport(
                acn=acn,
                date=f"{year}-{month:02d}-{day:02d}",
                aircraft_type=aircraft,
                phase_of_flight=phase,
                anomaly_type=anomaly,
                contributing_factors=factors,
                narrative=narrative,
                synopsis=f"Synthetic: {anomaly} during {phase} involving {aircraft} at {airport}",
                is_synthetic=True,
            ))

        return reports

    @staticmethod
    def _load_templates() -> list[str]:
        return [
            "I was operating {aircraft} as PIC on a Part 121 flight into {airport}. During {phase} "
            "at {altitude} feet in {weather} conditions, we experienced {anomaly}. The flight crew "
            "({callsign}) immediately initiated appropriate procedures per the QRH. ATC was notified "
            "and we were given priority handling. The situation was resolved without further incident "
            "but highlights the importance of crew coordination and adherence to standard procedures. "
            "Contributing factors included high workload during a critical phase of flight and degraded "
            "environmental conditions. I recommend enhanced training scenarios for this type of event.",

            "While serving as First Officer on {aircraft} flight {callsign}, we encountered {anomaly} "
            "during the {phase} phase at approximately {altitude} feet MSL. Weather was reported as "
            "{weather}. The Captain took controls and I handled communications with {airport} approach "
            "control. We declared PAN PAN and were vectored for an immediate approach. Maintenance was "
            "notified prior to landing. Post-flight inspection revealed the root cause. This event "
            "could have been prevented with more rigorous preflight inspection procedures. The rapid "
            "response of ATC at {airport} was commendable and contributed to a safe outcome.",

            "As a controller working {airport} TRACON, I observed {aircraft} ({callsign}) deviate from "
            "assigned altitude during {phase}. The aircraft was at {altitude} when the deviation "
            "occurred in {weather} conditions. I issued a traffic advisory and corrective instruction. "
            "The pilot reported {anomaly} as the cause of the deviation. I coordinated with adjacent "
            "sectors to ensure separation. The event was classified as a pilot deviation. Workload was "
            "moderate at the time. I believe better communication between flight crew and ATC could "
            "have prevented this situation from developing. Training on the specific {anomaly} scenario "
            "should be emphasized during recurrent training.",

            "During operation of {aircraft} on a Part 91 VFR flight near {airport}, I experienced "
            "{anomaly} while in {phase} at {altitude} feet AGL. Conditions were {weather}. As sole "
            "pilot ({callsign}), I immediately applied the emergency procedures from memory and "
            "declared an emergency with {airport} tower. The controller cleared all traffic and I "
            "was able to land without incident. Post-flight analysis identified multiple contributing "
            "factors. This report is filed to increase awareness of this scenario, particularly for "
            "pilots operating similar aircraft types in comparable conditions.",

            "Operating {aircraft} ({callsign}) on approach to {airport} runway in {weather} conditions, "
            "we encountered {anomaly} at approximately {altitude} feet during {phase}. The autopilot "
            "was disconnected and I hand-flew the aircraft for the remainder of the approach. The First "
            "Officer assisted with checklists and ATC coordination. We executed a go-around on the "
            "first attempt due to unstabilized approach criteria, then landed successfully on the "
            "second attempt. Fatigue was a factor as this was the fourth leg of a long duty day. I "
            "recommend the company review scheduling practices for high-density routes into {airport}.",
        ]


def process_reports(raw_reports: list[ASRSReport]) -> list[ProcessedReport]:
    """Clean and structure raw reports into processed format."""
    processed = []
    for report in raw_reports:
        cleaned_narrative = clean_narrative(report.narrative)
        if not cleaned_narrative or len(cleaned_narrative) < 50:
            continue

        text = f"{report.synopsis}\n\n{cleaned_narrative}" if report.synopsis else cleaned_narrative

        processed.append(ProcessedReport(
            id=report.acn,
            text=text,
            metadata={
                "acn": report.acn,
                "date": report.date,
                "aircraft_type": report.aircraft_type,
                "phase_of_flight": report.phase_of_flight,
                "anomaly_type": report.anomaly_type,
                "contributing_factors": report.contributing_factors,
                "is_synthetic": report.is_synthetic,
            },
        ))

    return processed


def save_reports(reports: list[ProcessedReport], path: Path) -> None:
    """Write processed reports to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for report in reports:
            f.write(json.dumps({"id": report.id, "text": report.text, "metadata": report.metadata}) + "\n")
    print(f"Saved {len(reports)} reports to {path}")


def load_reports(path: Optional[Path] = None) -> list[ProcessedReport]:
    """Load processed reports from JSONL."""
    if path is None:
        path = PROCESSED_DIR / "reports.jsonl"
    reports = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            reports.append(ProcessedReport(
                id=data["id"],
                text=data["text"],
                metadata=data.get("metadata", {}),
            ))
    return reports


def run_ingestion(target_count: int = 500) -> list[ProcessedReport]:
    """Run the full ingestion pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "reports.jsonl"
    if output_path.exists():
        existing = load_reports(output_path)
        if len(existing) >= target_count:
            print(f"Found {len(existing)} existing reports, skipping ingestion")
            return existing

    # Option A: try ASRS download
    print("Attempting ASRS database download...")
    downloader = ASRSDownloader()
    raw_data = downloader.download(target_count)

    if raw_data:
        raw_reports = []
        for item in raw_data:
            raw_reports.append(ASRSReport(
                acn=str(item.get("acn", "")),
                date=item.get("date", ""),
                aircraft_type=normalize_aircraft(item.get("aircraft_type", "")),
                phase_of_flight=item.get("phase_of_flight", extract_phase(item.get("narrative", ""))),
                anomaly_type=item.get("anomaly_type", extract_anomaly(item.get("narrative", ""))),
                contributing_factors=item.get("contributing_factors", []),
                narrative=item.get("narrative", ""),
                synopsis=item.get("synopsis", ""),
                is_synthetic=False,
            ))
        print(f"Downloaded {len(raw_reports)} reports from ASRS")
    else:
        # Option C: synthetic generation (skipping B — PDF parsing too unreliable)
        print("ASRS download failed. Generating synthetic reports...")
        generator = SyntheticGenerator()
        raw_reports = generator.generate_batch(target_count)
        print(f"Generated {len(raw_reports)} synthetic reports")

        # Save raw synthetic data
        raw_path = RAW_DIR / "synthetic_reports.jsonl"
        with open(raw_path, "w") as f:
            for r in raw_reports:
                f.write(json.dumps(asdict(r)) + "\n")

    processed = process_reports(raw_reports)
    save_reports(processed, output_path)
    return processed


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    reports = run_ingestion(target)
    print(f"\nIngestion complete: {len(reports)} processed reports")
