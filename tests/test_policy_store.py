from pathlib import Path

from paig.policy_store import load_policies

ROOT = Path(__file__).resolve().parents[1]


def test_loads_all_policy_files():
    chunks = load_policies(ROOT / "data" / "policies")
    assert len(chunks) > 0
    doc_ids = {c.doc_id for c in chunks}
    assert "POL-RAD-014" in doc_ids


def test_sections_are_split():
    chunks = load_policies(ROOT / "data" / "policies")
    lumbar_chunks = [c for c in chunks if c.doc_id == "POL-RAD-014"]
    headings = {c.heading for c in lumbar_chunks}
    assert "Medical Necessity Criteria" in headings
    assert "Documentation Requirements" in headings
