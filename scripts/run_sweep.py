"""Evaluation sweep over the synthetic sample.

Runs every ASN found in the sample folder through all five designs and writes a
per-case result JSON plus a summary table of dispositions. This is the public,
synthetic analogue of the sweep used in the dissertation: there, the same driver
was run over the full production, pre-integration and synthetic corpus (which is
confidential and is not published here). On the bundled synthetic sample it lets
anyone exercise the five designs end to end.

Stages 3 and 5 call a language model, so a configured ``.env`` is required
(copy ``.env.example`` to ``.env`` and fill in the keys).

Usage
-----
    python scripts/run_sweep.py
    python scripts/run_sweep.py --asn-dir sample_data --out results/sweep_sample
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

MODES = ["none", "deterministic", "llm", "tool_use", "partition"]
LABEL = {"none": "pure LLM", "deterministic": "post hoc", "llm": "LLM-as-judge",
         "tool_use": "in generation", "partition": "pre hoc"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asn-dir", default="sample_data",
                    help="folder of ASN cXML files to sweep (default: sample_data)")
    ap.add_argument("--out", default="results/sweep_sample",
                    help="output folder for per-case result JSON")
    ap.add_argument("--modes", default=",".join(MODES),
                    help="comma-separated subset of designs to run")
    args = ap.parse_args()

    asn_dir = (ROOT / args.asn_dir).resolve()
    # The pipeline resolves PO baselines from ASN_DATA_DIR; point it at the ASN folder.
    os.environ.setdefault("ASN_DATA_DIR", str(asn_dir))
    import pipeline as P  # noqa: E402  (import after env setup)

    out = (ROOT / args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # ASN files are those whose header is a ShipNotice (skip PO_*.xml baselines).
    asns = sorted(p for p in asn_dir.rglob("*.xml")
                  if not p.name.startswith("PO_")
                  and "ShipNoticeRequest" in p.read_text(encoding="utf-8"))
    if not asns:
        print(f"No ASN files found under {asn_dir}")
        return

    table = []
    for asn_path in asns:
        asn_xml = asn_path.read_text(encoding="utf-8")
        row = {"case": asn_path.stem}
        for mode in modes:
            res = P.run_pipeline_from_xml(asn_xml, name=asn_path.stem, enforcer_mode=mode)
            (out / f"{asn_path.stem}__{mode}.json").write_text(
                json.dumps(res, indent=2, default=str), encoding="utf-8")
            row[mode] = res.get("disposition", "ERROR")
        table.append(row)

    # Print a compact disposition table.
    width = max(len(r["case"]) for r in table)
    header = "case".ljust(width) + "".join(f"  {LABEL[m]:>13s}" for m in modes)
    print(header)
    print("-" * len(header))
    for r in table:
        print(r["case"].ljust(width) + "".join(f"  {r[m]:>13s}" for m in modes))
    print(f"\nPer-case results written to {out}")


if __name__ == "__main__":
    main()
