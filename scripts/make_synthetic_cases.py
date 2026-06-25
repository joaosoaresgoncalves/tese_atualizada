"""Deterministic synthetic-case generator (mutation testing).

Mutates the clean synthetic ASN in ``sample_data/asn_clean.xml`` to target one
rule at a time, mirroring the method used to build the synthetic evaluation
block of the dissertation (Section 4.5.5): a clean, schema-valid seed is altered
by a single deterministic edit, so the expected disposition follows by
construction. Every identifier is synthetic; no real data is read or written.

This is a compact demonstration of the method, not the full 51-mutant matrix.

Usage
-----
    python scripts/make_synthetic_cases.py --out sample_data/generated
"""
import argparse
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "sample_data" / "asn_clean.xml"

# name -> (one-line description, mutation function)
MUTANTS = {
    "MUT-R01_overship": "Quantity 100 -> 130 (over PO band, 0% tol) | R01 CRITICAL -> REJECT",
    "MUT-R08_currency": "Line currency EUR -> USD                   | R08 CRITICAL -> REJECT",
    "MUT-R19_longid":   "shipmentID padded beyond 35 characters     | R19 CRITICAL -> REJECT",
    "MUT-R03_latedate": "deliveryDate shifted +45 days              | R03 WARNING  -> REVIEW",
}


def _load():
    parser = etree.XMLParser(remove_blank_text=False)
    return etree.parse(str(CLEAN), parser)


def _header(tree):
    return tree.find(".//ShipNoticeHeader")


def _first_item(tree):
    return tree.find(".//ShipNoticeItem")


def build(name, tree):
    _header(tree).set("shipmentID", f"SHIP_SYN_{name.replace('-', '_')}")
    if name == "MUT-R01_overship":
        _first_item(tree).set("quantity", "130")
    elif name == "MUT-R08_currency":
        _first_item(tree).find(".//Money").set("currency", "USD")
    elif name == "MUT-R19_longid":
        _header(tree).set("shipmentID", "SHIP_SYN_" + "X" * 40)  # length > 35
    elif name == "MUT-R03_latedate":
        _header(tree).set("deliveryDate", "2026-06-16T00:00:00+00:00")
    return tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "sample_data" / "generated"),
                    help="output folder for the generated mutants")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, desc in MUTANTS.items():
        tree = build(name, _load())
        path = out / f"{name}.xml"
        tree.write(str(path), xml_declaration=True, encoding="UTF-8")
        print(f"  {path.name:24s} {desc}")
    print(f"\n{len(MUTANTS)} synthetic mutants written to {out}")


if __name__ == "__main__":
    main()
