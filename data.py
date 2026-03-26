from datasets import load_dataset
from config import DATASET_NAME

METADATA_FIELDS = [
    "ACN", "Date", "Local_Time_Of_Day", "Place", "State_Reference",
    "Altitude_MSL", "Flight_Phase", "Aircraft_Operator", "Aircraft_Make_Model",
    "Primary_Problem", "Contributing_Factors", "Human_Factors",
    "Person_Function", "Person_Flight_Activity",
]


def load_reports():
    ds = load_dataset(DATASET_NAME, split="train")
    return ds


def format_metadata(report: dict) -> str:
    parts = []
    for field in METADATA_FIELDS:
        value = report.get(field)
        if value and str(value).strip() and str(value).strip().lower() != "nan":
            label = field.replace("_", " ")
            parts.append(f"{label}: {value}")
    return "\n".join(parts)


def chunk_report(report: dict) -> dict | None:
    """Build a single retrievable chunk from one ASRS report.

    Combines narrative, synopsis, callback, and structured metadata
    into one text block. Returns None if the report has no usable text.
    """
    acn = report.get("ACN", "unknown")
    narrative = (report.get("Report_1_Narrative") or "").strip()
    synopsis = (report.get("Synopsis") or "").strip()
    callback = (report.get("Report_1_Callback_Narrative") or "").strip()

    if not narrative and not synopsis:
        return None

    metadata = format_metadata(report)

    sections = [f"ASRS Report {acn}"]
    if metadata:
        sections.append(metadata)
    if narrative:
        sections.append(f"Narrative: {narrative}")
    if synopsis:
        sections.append(f"Synopsis: {synopsis}")
    if callback:
        sections.append(f"Callback: {callback}")

    return {
        "acn": str(acn),
        "text": "\n\n".join(sections),
    }


def chunk_all_reports(dataset) -> list[dict]:
    chunks = []
    for report in dataset:
        chunk = chunk_report(report)
        if chunk:
            chunks.append(chunk)
    return chunks
