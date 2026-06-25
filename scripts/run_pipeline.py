"""Run the validation pipeline on a single ASN (smoke test).

Reads a cXML ASN, finds the matching PO baseline in ``sample_data/`` (or the
folder named by the ``ASN_DATA_DIR`` environment variable), runs the seven-stage
pipeline under one of the five designs, and prints the resulting disposition and
per-rule findings as JSON.

Stages 3 and 5 call a language model, so this needs a configured ``.env``
(copy ``.env.example`` to ``.env`` and fill in the keys). The figure notebook,
by contrast, needs no key.

Examples
--------
    python scripts/run_pipeline.py sample_data/asn_clean.xml --mode partition
    python scripts/run_pipeline.py sample_data/asn_overship.xml --mode deterministic

Design keys: none (pure LLM), deterministic (post hoc), llm (LLM-as-judge),
tool_use (in generation), partition (pre hoc).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
# Point the pipeline at the bundled synthetic sample unless the caller overrides it.
os.environ.setdefault("ASN_DATA_DIR", str(ROOT / "sample_data"))

import pipeline as P  # noqa: E402  (import after sys.path / env setup)

MODES = ["none", "deterministic", "llm", "tool_use", "partition"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("asn", help="path to an ASN cXML file")
    ap.add_argument("--mode", default="partition", choices=MODES,
                    help="enforcer design to run (default: partition / pre hoc)")
    args = ap.parse_args()

    asn_xml = Path(args.asn).read_text(encoding="utf-8")
    result = P.run_pipeline_from_xml(asn_xml, name=Path(args.asn).stem,
                                     enforcer_mode=args.mode)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
