# Hybrid LLM + deterministic pipeline for procurement document validation

This repository accompanies my master's dissertation, *Hybrid LLM Deterministic
Pipeline for Procurement Document Validation* (FEUP, M.EGI). It contains the
validation pipeline, the 22-rule register, the five pipeline designs compared in
the study, and the aggregated, anonymised results behind every figure in the
results chapter. The figures regenerate offline from the committed result tables
with no API key; the pipeline itself can be run on a small synthetic sample if
you supply your own model credentials.

The task is to validate an Advanced Shipping Notice (ASN) against its originating
Purchase Order (PO). A language model reads the supplier-specific fields and the
contract terms, and a deterministic **enforcer** holds authority over every rule
whose verdict can be settled by computation. The dissertation benchmarks five
designs that differ in whether the enforcer is present and where it acts.

## Reproduce the figures in 3 steps (no API key)

```bash
git clone https://github.com/joaosoaresgoncalves/tese_atualizada.git
cd tese_atualizada
pip install -r requirements.txt
jupyter notebook notebooks/figures.ipynb     # run all cells, top to bottom
```

The notebook reads the aggregated result tables in `results/` and regenerates the
data figures of the results chapter: disposition metrics, per-rule recall, the
paired effect sizes, the cost/accuracy Pareto, the latency spread, the confusion
matrices, the cross-model comparison and the supplier-conditioning ablation. The
final cell prints a spot check against the headline numbers in the dissertation
(buyer accuracy 0.81 without the enforcer vs 0.88 with it; EUR 0.073 per ASN and
7.89 s median for the recommended pre hoc design; the cross-model accuracies
0.69 / 0.84 / 0.72 / 0.81 / 0.81).

## Re-run the pipeline on the sample (optional, needs a model)

Stages 3 and 5 of the pipeline call a language model, so this step needs
credentials.

```bash
cp .env.example .env          # then fill in your Azure OpenAI key and endpoint
python scripts/run_pipeline.py sample_data/asn_clean.xml --mode partition
python scripts/run_pipeline.py sample_data/asn_overship.xml --mode deterministic
```

Run all five designs over the synthetic sample and print a disposition table:

```bash
python scripts/run_sweep.py
```

Generate a few synthetic mutants by deterministic edits to the clean sample
(the method used to build the synthetic evaluation block; no model needed):

```bash
python scripts/make_synthetic_cases.py
```

The five design keys are `none` (pure LLM), `deterministic` (post hoc),
`llm` (LLM-as-judge), `tool_use` (in generation) and `partition` (pre hoc).

## Repository layout

```
pipeline/
  pipeline.py          seven-stage pipeline: parse PO, ingest ASN, normalise,
                       build context, evaluate rules, enforce, reduce to a verdict
  RULES.md             the 22-rule register and the five designs
  validation logic     the enforcer, the rule classifiers and the prompts live in pipeline.py
scripts/
  run_pipeline.py      run one ASN through one design (needs a model)
  run_sweep.py         run the sample through all five designs (needs a model)
  make_synthetic_cases.py   deterministic synthetic-case generator (no model)
results/
  metrics_buyer.json        disposition metrics + confusion + recall, 32 buyer cases
  metrics_synthetic.json    disposition metrics + recall + multi-rule, 67 synthetic cases
  confusion.json            per-design confusion matrices, both cohorts
  operational.json          cost and latency spread per design
  cross_model_buyer.json    primary vs second model, buyer cohort
  cross_model_synthetic.json  second model, synthetic cohort
  contrasts.json            paired effect sizes vs the pre hoc design
  supplier_ablation.json    RQ3 supplier-conditioning ablation
  reproducibility.json      verdict stability across repeated runs
  design_map.json           mode key -> design name -> enforcer flag
sample_data/
  PO_ORDER_SYN_0001.xml     a synthetic PO baseline
  asn_clean.xml             a clean ASN (expected PASS)
  asn_overship.xml          an over-shipped ASN (expected REJECT on R01)
notebooks/
  figures.ipynb        regenerates every data figure from results/
```

## Data and anonymisation

The pipeline was developed and evaluated on confidential procurement documents
from the partner organisation. **None of that data is in this repository.** The
real ASN, PO and goods-receipt documents, the buyer ground-truth annotations,
the supplier history corpus and the integration guidelines are all withheld.

What is published instead:

- **Aggregated results only.** The files in `results/` hold per-design summary
  statistics — accuracies, confusion counts, recall fractions, cost and latency
  summaries. They contain no document, supplier or order identifiers.
- **A fully synthetic sample.** Every identifier in `sample_data/` is invented
  (`AN_SYN_0001`, `ORDER_SYN_0001`, and so on). No supplier is named.

Because the evaluation corpus is confidential, the full sweep that produced the
results cannot be re-run from this repository. The figures are therefore
reproduced from the committed aggregates, and the pipeline is exercised on the
synthetic sample. Suppliers are referred to only as S1, S2, … in line with the
dissertation.

## Citation

If you refer to this work, please cite the dissertation: João Gonçalves,
*Hybrid LLM Deterministic Pipeline for Procurement Document Validation*,
Faculty of Engineering, University of Porto, 2026.
