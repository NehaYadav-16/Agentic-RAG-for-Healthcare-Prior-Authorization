from pathlib import Path

from paig.policy_store import load_policies
from paig.retriever import TfidfRetriever

ROOT = Path(__file__).resolve().parents[1]


def test_retrieves_relevant_policy_for_query():
    chunks = load_policies(ROOT / "data" / "policies")
    retriever = TfidfRetriever(chunks)
    results = retriever.retrieve("knee locking catching meniscal tear arthroscopy", top_k=3)
    assert len(results) == 3
    top_doc_ids = {r.chunk.doc_id for r in results}
    assert "POL-ORTHO-022" in top_doc_ids


def test_category_filter_restricts_results():
    chunks = load_policies(ROOT / "data" / "policies")
    retriever = TfidfRetriever(chunks)
    results = retriever.retrieve("medical necessity", top_k=10, procedure_category="bariatric_surgery")
    assert all(r.chunk.procedure_category == "bariatric_surgery" for r in results)
