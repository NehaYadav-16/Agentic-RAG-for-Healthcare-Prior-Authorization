# PAIG — Prior Authorization Intelligence & Grounding

**Agentic RAG for prior-authorization decisioning and denial-appeal drafting, grounded in payer medical policy documents.**

Health insurers and providers in the US spend billions of dollars a year on manual
prior-authorization (PA) review and claim-denial appeals. PAIG is a working,
fully-offline prototype of a retrieval-augmented, citation-grounded system that:

1. **Retrieves** the exact payer policy clauses relevant to a requested procedure.
2. **Checks** the claim against those clauses and produces an auditable, citation-backed checklist.
3. **Scores denial risk** with a trained ML model *before* the claim is submitted.
4. **Decides** likely-approved vs. high-denial-risk.
5. **Drafts a grounded appeal letter** — citing only the retrieved policy text, never inventing facts — when risk is high.

Every step is logged to an audit trail, which is the property regulators, UM
reviewers, and auditors actually need from an automated healthcare decision
system (see EU AI Act high-risk system requirements and US state prior-auth
reform laws, which both increasingly require explainability and human review).

> ⚠️ **This is a research/demo project built entirely on synthetic data.**
> It is not a medical device, not legal advice, and not validated against any
> real payer's actual policies. Any real deployment requires a licensed
> clinical/UM reviewer in the loop and legal review of the underlying policies.

## Why this problem

- US: the AHA and CAQH Index report tens of billions of dollars a year in
  administrative cost tied to prior authorization and claim denial/appeal
  cycles, and algorithmic-denial practices at major payers have drawn
  regulatory and legal scrutiny (2023–2025).
- EU: the European Health Data Space Regulation and national HTA bodies
  (e.g., NICE in the UK) create an equivalent policy-grounded coverage
  decision problem, and the EU AI Act classifies many healthcare decision
  systems as "high-risk," requiring exactly the kind of grounded citation
  trail this project produces.
- Both regions are converging on the same requirement: **AI-assisted
  decisions must be explainable and traceable to source policy**, not a
  black box. That's the design constraint this project is built around.

## Architecture

```
                 ┌───────────────────────┐
                 │   Payer policy docs    │   (data/policies/*.md)
                 │  (medical necessity,   │
                 │  documentation, etc.)  │
                 └───────────┬────────────┘
                             │ chunk by section
                             ▼
                 ┌───────────────────────┐
                 │   TF-IDF Retriever     │   src/paig/retriever.py
                 │ (swap for dense/vector │
                 │  DB in production)     │
                 └───────────┬────────────┘
                             │ top-k cited chunks
                             ▼
   claim  ──────▶ ┌───────────────────────┐
  (structured  )   │   Criteria Checker     │   src/paig/criteria_checker.py
                 │ (claim fields × policy)│
                 └───────────┬────────────┘
                             │ checklist + citations
                             ▼
                 ┌───────────────────────┐
  claim  ──────▶ │ Denial-Risk Classifier │   src/paig/denial_model.py
  (structured)    │ (Gradient Boosting)    │
                 └───────────┬────────────┘
                             │ risk score
                             ▼
                 ┌───────────────────────┐
                 │   Decision + Appeal    │   src/paig/pipeline.py
                 │   Letter Drafting      │   src/paig/appeal_generator.py
                 └───────────┬────────────┘
                             │
                             ▼
                 Audit trail · Decision · Citations · Appeal draft
```

This multi-step retrieve → check → score → decide → draft chain is the
"agent": each step is a small, independently testable function, and the
full reasoning trace is returned with every result (`PriorAuthResult.audit_trail`).

## Repo layout

```
prior-auth-rag/
├── app.py                     # Streamlit demo UI
├── data/
│   ├── policies/*.md          # 10 synthetic payer medical-necessity policies
│   └── claims/*.csv           # generated synthetic claims dataset
├── models/                    # trained model + evaluation report (generated)
├── scripts/
│   ├── generate_data.py       # builds the synthetic claims dataset
│   ├── train_model.py         # trains + saves the denial-risk classifier
│   ├── evaluate_rag.py        # retrieval + end-to-end + $ savings evaluation
│   └── demo_cli.py            # runs sample claims through the full agent
├── src/paig/                  # the package
│   ├── policy_store.py        # loads & chunks policy markdown files
│   ├── retriever.py           # TF-IDF retriever
│   ├── data_gen.py            # synthetic claims generator
│   ├── denial_model.py        # ML denial-risk classifier
│   ├── criteria_checker.py    # claim × policy criteria matching
│   ├── appeal_generator.py    # grounded letter drafting (template + optional LLM)
│   ├── cost_savings.py        # $ savings estimator
│   └── pipeline.py            # orchestrating agent
└── tests/                     # pytest suite (9 tests, all passing)
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_data.py     # builds data/claims/claims_synthetic.csv
python scripts/train_model.py       # trains models/denial_model.joblib
python scripts/demo_cli.py          # runs 2 sample claims end-to-end
python scripts/evaluate_rag.py      # retrieval + pipeline + $ savings report

streamlit run app.py                # interactive demo UI
pytest -v                           # 9 tests, all should pass
```

No API key is required. The appeal-letter generator defaults to a fully
deterministic, offline template backend. If `ANTHROPIC_API_KEY` is set and
the `anthropic` package is installed, it automatically switches to Claude to
rephrase the *same grounded facts* into more natural prose under a strict
no-new-facts system prompt — demonstrating the "optional LLM polish on top of
a deterministic, auditable core" pattern that's appropriate for regulated
use cases.

## Results (from this repo, reproducible via the commands above)

| Metric | Value |
|---|---|
| Retrieval Precision@1 (10-query gold set) | 100% |
| Retrieval Precision@3 (10-query gold set) | 100% |
| Denial-risk model ROC-AUC (held-out test set) | 0.80 |
| Denial-risk model precision / recall | 0.62 / 0.49 |
| High-risk flag rate on a held-out sample of 500 claims | 20.4% (true denial rate: ~27%) |

### Illustrative cost-savings projection

Using configurable, labeled assumptions in `src/paig/cost_savings.py`
(25 min manual review/claim, 45 min manual appeal drafting, $32/hr staff
cost, 20% appeal rate, 70% time reduction from automation — replace with
your own time-and-motion data before using for real budgeting):

| Volume | Manual cost/month | Automated cost/month | Savings/month |
|---|---|---|---|
| 10,000 claims | $181,333 | $54,400 | **$126,933 (70%)** |

## For research / publication

This is designed to extend into a short paper (e.g., an arXiv preprint or a
workshop submission to a clinical-NLP venue such as ACL ClinicalNLP or AMIA):

- **Novelty angle**: most prior-auth automation literature focuses on denial
  *prediction*; this combines prediction with citation-grounded retrieval
  and appeal-letter generation in one auditable pipeline, directly
  addressing the explainability requirements payers/regulators are now
  demanding (EU AI Act, US NAIC model bulletins on AI in claims).
- **Evaluation to extend**: replace the heuristic criteria checker with an
  LLM-based entailment check (claim note → does it satisfy criterion X?)
  and report factuality/citation-accuracy against a human-labeled set —
  this is the natural "Experiments" section for a paper.
- **Baselines to compare against**: rules-only system (no ML risk model),
  ML-only system (no retrieval/citations), and this full pipeline — ablation
  table showing the retrieval+citation layer's contribution to reviewer trust
  and appeal-overturn rate would be a strong empirical result.
- **Datasets to extend to real data (with proper access/IRB)**: MIMIC-IV for
  clinical note grounding, CMS public claims data, publicly posted payer
  medical policy PDFs (replacing the synthetic ones here).

## License

MIT — see [LICENSE](LICENSE).
