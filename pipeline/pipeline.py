"""
PO-ASN Validation Pipeline — Standalone Module
Extracted from LLM_AI_Files_Workflow.ipynb

7-stage pipeline:
  1. PO Baseline Capture (lxml)
  2. ASN Ingestion (lxml)
  3. LLM Parsing & Normalization (Azure OpenAI)
  4. Prompt Builder & Context Integrator
  5. LLM Validation Agent (Azure OpenAI)
  6. Output Enforcer
  7. Decision Router + Audit
"""

import os, json, re, pathlib
from datetime import datetime, timedelta, timezone
from lxml import etree
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

# ── Load environment (look in parent dir too for .env) ──
BASE = pathlib.Path(__file__).resolve().parent
load_dotenv(BASE.parent / ".env")
load_dotenv()  # also check local dir

# ── Model backend switch ──────────────────────────────────────────────
# Default: Azure OpenAI (the thesis baseline). Set LLM_BACKEND=openai_compat to
# target an OpenAI-compatible endpoint such as Mistral-Large-3 on Azure AI
# Foundry (cross-model robustness replication). ONLY the transport changes;
# prompts, rules and the enforcer logic are untouched.
_LLM_BACKEND = os.getenv("LLM_BACKEND", "azure_openai").lower()
if _LLM_BACKEND == "openai_compat":
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", ""),   # Foundry OpenAI-compatible URL
        api_key=os.getenv("LLM_API_KEY", ""),
        timeout=300.0,
    )
    DEPLOYMENT = os.getenv("LLM_MODEL", "mistral-large-3")
else:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        timeout=300.0,  # 5 minutes — large multi-PO ASNs need more time
    )
    DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "mistral")

# Capability flags — providers differ on `seed` and `json_schema` support.
# For Mistral on Foundry, set LLM_SUPPORTS_SEED=false and LLM_SUPPORTS_JSON_SCHEMA=false.
_LLM_SUPPORTS_SEED = os.getenv("LLM_SUPPORTS_SEED", "true").lower() == "true"
_LLM_SUPPORTS_JSON_SCHEMA = os.getenv("LLM_SUPPORTS_JSON_SCHEMA", "true").lower() == "true"
# In-generation (tool_use): force a tool call on the FIRST turn so the model
# actually exercises the deterministic tools — some models (e.g. Mistral) skip
# them under tool_choice="auto". Later turns use "auto" so the model can finish.
# Default off to preserve the gpt baseline behaviour; set TOOL_USE_FORCE_FIRST=true.
_TOOL_USE_FORCE_FIRST = os.getenv("TOOL_USE_FORCE_FIRST", "false").lower() == "true"
# Some providers (Mistral) reject `tools` together with response_format=json_object
# ("Cannot use json response type with tools"). When set, drop response_format in
# the tool_use loop; the model still returns tool calls and a JSON answer in the
# message content (the prompt instructs JSON), which is parsed downstream. Default
# off to preserve the gpt baseline (which enforces json_object server-side).
_LLM_TOOLS_NO_RESPONSE_FORMAT = os.getenv("LLM_TOOLS_NO_RESPONSE_FORMAT", "false").lower() == "true"

# ── Enforcer mode (thesis RQ1 ablation) ──
# "none"           -- pure LLM Stage 5, no Stage 6 enforcer
# "deterministic"  -- LLM Stage 5 + Python Stage 6 enforcer (default)
# "llm"            -- LLM Stage 5 + LLM-as-audit Stage 6 reviewing all 22
#                    rules (no scope filter). json_object response format.
# "tool_use"       -- LLM Stage 5 with in-generation function calling
#                    over arithmetic verification tools; no Stage 6 enforcer.
#                    Note: R18 (duplicate detection) cannot be expressed as
#                    a stateless tool and is therefore not covered.
# "partition"      -- Stage 5 split: 5a deterministic verdicts for 8 owned
#                    rules (R01,R02,R03,R06,R15,R16,R18,R19); 5b LLM-narrow
#                    over the 14 remaining rules with 5a's verdicts embedded
#                    as context. Stage 6 merges the 8 + 14 verdicts and
#                    applies severity capping and rule injection on the 14
#                    LM-regime verdicts only; the 8 deterministic verdicts
#                    from Stage 5a are not modified. Output: exactly 22
#                    ruleResult entries.
ENFORCER_MODE = os.getenv("ENFORCER_MODE", "deterministic")

# Best-effort determinism: passed to every chat.completions.create() call
# across all four enforcer modes. Combined with temperature=0.0 this makes
# Azure OpenAI reproduce identical outputs across reruns up to backend
# sampling noise. Azure does not guarantee bit-for-bit determinism even
# with seed -- check system_fingerprint on responses to detect a backend
# rotation.
DET_SEED = int(os.getenv("DET_SEED", "42"))

# Rules the deterministic enforcer arithmetically re-verifies or injects.
_DETERMINISTIC_OWNED_RULES = ("R01", "R02", "R03", "R06", "R15", "R16", "R18", "R19")

# Per-supplier prompt-conditioning ("simple viable RAG"). When enabled, Stage 4
# loads a YAML profile for the supplier identified in the ASN/PO and injects a
# one-line "historical note" into the system prompt. Profiles are generated by
# Metodologia/scripts/build_supplier_profiles.py from past sweep_results.
USE_SUPPLIER_PROFILES = os.getenv("USE_SUPPLIER_PROFILES", "false").lower() == "true"
SUPPLIER_PROFILES_DIR = BASE / "data" / "supplier_profiles"

# RAG_PLACEBO: placebo arm for the supplier-conditioning ("simple RAG") ablation.
# When ON (and USE_SUPPLIER_PROFILES is ON), the Stage-4 lookup still fires and
# still injects a "SUPPLIER HISTORY (<real vendor id>): …" line — but the history
# *content* is taken from a DIFFERENT supplier's profile, chosen deterministically.
# The prompt framing is byte-for-byte identical to the real arm; only the facts are
# wrong. This isolates whether the model uses the *specific* supplier history or
# merely the *presence* of an authoritative-looking history sentence. A null
# difference between the real and placebo arms means the gain (if any) is framing,
# not retrieval. See Metodologia/scripts/run_evaluation_sweep.py --rag-placebo.
RAG_PLACEBO = os.getenv("RAG_PLACEBO", "false").lower() == "true"

# ── Stage-level LLM experiment flags (thesis iter3) ──
# USE_STAGE2_SCHEMA:   Stage 2 ASN parse uses json_schema (structured outputs).
#                      Default ON -- low-risk fix for string/number drift bugs.
# USE_STAGE5_FEW_SHOT: Stage 5 validation prompt includes 3 worked-example
#                      (input, correct ruleResults) pairs. Default ON.
# USE_STAGE5_SCHEMA:   Stage 5 validation uses json_schema on its output.
#                      Default OFF -- keep as ablation variable (large output,
#                      semantic fields include free text, unknown interaction
#                      with reasoning quality).
USE_STAGE2_SCHEMA   = os.getenv("USE_STAGE2_SCHEMA",   "true").lower() == "true"
USE_STAGE5_FEW_SHOT = os.getenv("USE_STAGE5_FEW_SHOT", "true").lower() == "true"
USE_STAGE5_SCHEMA   = os.getenv("USE_STAGE5_SCHEMA",   "false").lower() == "true"
# DISABLE_R18: neutralise the stateful duplicate-ASN check (R18). R18 cannot be
# evaluated in a single-file / repeated-run setup — it remembers ASNs across
# runs via seen_asns.json, so re-processing the corpus makes every ASN look
# "seen before" and fire R18 spuriously. Set true for disposition evaluation.
DISABLE_R18         = os.getenv("DISABLE_R18",         "false").lower() == "true"

# ── Paths ──
# Folder holding the PO baselines (and case XML) the pipeline reads. Defaults to
# the bundled synthetic sample; override with the ASN_DATA_DIR environment variable.
DATA = pathlib.Path(os.getenv("ASN_DATA_DIR", str(BASE.parent / "sample_data")))
OUTPUT = BASE / "output"
for sub in ["baselines", "parsed", "reports"]:
    (OUTPUT / sub).mkdir(parents=True, exist_ok=True)


def _safe_filename_id(s: str) -> str:
    # Some Ariba networks (notably AT-cluster suppliers) issue shipmentIDs of
    # the form "201/3464679", which Path() interprets as a directory separator
    # on disk. Sanitise to keep filename semantics intact while preserving
    # readability. R18/R19 rule logic still sees the original shipmentID.
    return (s or "unknown").replace("/", "_").replace("\\", "_")


def _parse_money(s, default: float = 0.0) -> float:
    """Robust numeric parsing for cXML monetary / quantity values.

    cXML payloads come from heterogeneous suppliers and carry numeric
    strings in mixed locale formats. Plain `float(s)` blows up on:
        "1,601.57"   (US-style: comma = thousand sep, dot = decimal)
        "1.601,57"   (EU-style: dot = thousand sep, comma = decimal)
        "1,57"       (EU-style: comma = decimal-only)
    This caused PO_ORDER_ID_0080 (P-40) to fail Stage 1 ingestion.

    Heuristic:
      - If both '.' and ',' appear, the LAST one is the decimal separator
        and the OTHER one is the thousand-separator (strip it out).
      - If only ',' appears, treat 3-digit suffix as thousand sep
        ("1,000" -> 1000) and 1- or 2-digit suffix as decimal
        ("1,57" -> 1.57).
      - Strip leading/trailing whitespace + currency symbols (€/$/£) before
        parsing.
    """
    if s is None:
        return default
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip()
    if not txt:
        return default
    # Strip common currency prefixes/suffixes
    for sym in ("€", "$", "£", "¥", "EUR", "USD", "GBP", "CHF"):
        txt = txt.replace(sym, "")
    txt = txt.strip()
    if not txt:
        return default
    has_dot = "." in txt
    has_comma = "," in txt
    if has_dot and has_comma:
        # Last separator is the decimal one; strip the other
        if txt.rfind(",") > txt.rfind("."):
            txt = txt.replace(".", "").replace(",", ".")
        else:
            txt = txt.replace(",", "")
    elif has_comma:
        # Single-comma case: 3-digit suffix → thousand sep; else decimal
        i = txt.rfind(",")
        suffix = txt[i + 1:]
        if len(suffix) == 3 and suffix.isdigit():
            txt = txt.replace(",", "")
        else:
            txt = txt.replace(",", ".")
    # has_dot only, or no separator: float() handles directly
    try:
        return float(txt)
    except ValueError:
        return default


# UOM equivalence map — UN/CEFACT codes that mean the same physical unit.
# Used by the R05 LLM prompt and (when the deterministic R05 check is
# enabled) the enforcer's same-canonical-class comparison.
_UOM_EQUIVALENCE = {
    # "Piece" / "Each" cluster — UN/CEFACT H87 ≡ PCE ≡ EA ≡ PC
    "PCE": "PIECE", "H87": "PIECE", "EA": "PIECE", "PC": "PIECE",
    "C62": "PIECE",  # "one"
    # Mass cluster
    "KGM": "KG", "KG": "KG",
    "GRM": "GRAM", "G": "GRAM",
    "LBR": "POUND", "LB": "POUND",
    "TNE": "TONNE", "T": "TONNE",
    # Length cluster
    "MTR": "METRE", "M": "METRE",
    "CMT": "CM", "CM": "CM",
    "MMT": "MM", "MM": "MM",
    "INH": "INCH", "IN": "INCH",
    "FOT": "FOOT", "FT": "FOOT",
    # Volume cluster
    "LTR": "LITRE", "L": "LITRE",
    "MLT": "ML", "ML": "ML",
    "MTQ": "M3",
    # Area cluster
    "MTK": "M2",
    # Time
    "HUR": "HOUR", "HR": "HOUR", "H": "HOUR",
}


def _uom_canonical(uom: str) -> str:
    """Map a UOM string to its canonical class (e.g. 'PCE' / 'H87' → 'PIECE').
    Unknown UOMs are returned uppercase-unchanged so genuine novelties are
    still distinguishable."""
    if not uom:
        return ""
    key = uom.strip().upper()
    return _UOM_EQUIVALENCE.get(key, key)

# ── Duplicate tracker (persisted across restarts) ──
_SEEN_ASNS_FILE = OUTPUT / "seen_asns.json"

def _load_seen_asns() -> set:
    if _SEEN_ASNS_FILE.exists():
        try:
            data = json.loads(_SEEN_ASNS_FILE.read_text())
            return {tuple(item) for item in data}
        except Exception:
            pass
    return set()

def _save_seen_asns(seen: set):
    _SEEN_ASNS_FILE.write_text(json.dumps([list(item) for item in seen], indent=2), encoding="utf-8")

_seen_asns = _load_seen_asns()


# ══════════════════════════════════════════════════════════════
# Helper: call LLM
# ══════════════════════════════════════════════════════════════

def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.0,
             schema: dict | None = None, schema_name: str = "response") -> str:
    """Send a chat completion request to Azure OpenAI and return the response text.

    If `schema` is provided, uses Azure structured outputs (`json_schema` mode)
    which enforces the schema server-side. Otherwise falls back to the looser
    `json_object` mode, which only guarantees valid JSON (not schema conformance).
    """
    try:
        if schema is not None and _LLM_SUPPORTS_JSON_SCHEMA:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            # No schema, or backend lacks json_schema support (e.g. Mistral on
            # Foundry): fall back to json_object (valid JSON, not schema-enforced).
            response_format = {"type": "json_object"}
        _kwargs = dict(
            model=DEPLOYMENT,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        if _LLM_SUPPORTS_SEED:
            _kwargs["seed"] = DET_SEED  # dropped for backends without seed support
        response = client.chat.completions.create(**_kwargs)
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [LLM ERROR] {type(e).__name__}: {e}")
        return json.dumps({
            "error": str(e),
            "alignment": {"portions": []},
            "ruleResults": [],
            "semanticFindings": [],
            "overallAssessment": f"LLM call failed: {type(e).__name__}: {e}",
        })


# ══════════════════════════════════════════════════════════════
# Stage 5 — In-generation tool definitions ("tool_use" ablation)
# ══════════════════════════════════════════════════════════════
# Stateless arithmetic-verification tools the Stage 5 LLM can call mid-generation
# under ENFORCER_MODE="tool_use". Each tool wraps the same closed-form check the
# deterministic enforcer applies in Stage 6, so the substitution is apples-to-
# apples on the deterministic-owned rule subset.
#
# R18 (duplicate detection) is structurally incapable of being a stateless tool
# call; tool_use mode does not address it. This is a finding to surface in the
# results.

def _tool_verify_quantity_match(asn_qty: float, po_qty: float,
                                 lower_pct: float, upper_pct: float) -> dict:
    """R01 (Shipped Quantity Match): asymmetric tolerance band
    [po_qty * (1 - lower_pct/100), po_qty * (1 + upper_pct/100)].
    Three-way classification (matches the deterministic enforcer)."""
    lower = po_qty * (1 - lower_pct / 100.0)
    upper = po_qty * (1 + upper_pct / 100.0)
    deviation = asn_qty - po_qty
    if asn_qty > upper:
        verdict, severity, status = "FAIL", "CRITICAL", "over_ship"
        explanation = f"ASN qty {asn_qty} exceeds upper bound {upper:.4g} (PO {po_qty}, +{upper_pct}%)."
    elif asn_qty < lower:
        verdict, severity, status = "PASS", "INFO", "under_ship"
        explanation = f"ASN qty {asn_qty} below lower bound {lower:.4g}; under-ship -> R16 escalation."
    else:
        verdict, severity, status = "PASS", "INFO", "in_range"
        explanation = f"ASN qty {asn_qty} within [{lower:.4g}, {upper:.4g}]."
    return {"rule_id": "R01", "verdict": verdict, "severity": severity,
            "status": status, "deviation": deviation,
            "lower_bound": lower, "upper_bound": upper,
            "explanation": explanation}


def _tool_verify_price_match(asn_price: float, po_price: float,
                              lower_pct: float, upper_pct: float) -> dict:
    """R02 (Unit Price Match): asymmetric tolerance band
    [po_price * (1 - lower_pct/100), po_price * (1 + upper_pct/100)]."""
    lower = po_price * (1 - lower_pct / 100.0)
    upper = po_price * (1 + upper_pct / 100.0)
    deviation = asn_price - po_price
    ok = lower <= asn_price <= upper
    verdict = "PASS" if ok else "FAIL"
    severity = "INFO" if ok else "CRITICAL"
    explanation = (f"ASN price {asn_price} {'within' if ok else 'outside'} "
                   f"[{lower:.4g}, {upper:.4g}] (PO {po_price}, -{lower_pct}%/+{upper_pct}%).")
    return {"rule_id": "R02", "verdict": verdict, "severity": severity,
            "deviation": deviation, "lower_bound": lower, "upper_bound": upper,
            "explanation": explanation}


def _tool_verify_line_total(unit_price: float, quantity: float, claimed_total: float) -> dict:
    """R15 (Total Value Reconciliation): unit_price * quantity == claimed_total
    within tolerance max(0.01, expected * 0.0001)."""
    expected = unit_price * quantity
    tol = max(0.01, expected * 0.0001)
    diff = abs(expected - claimed_total)
    ok = diff < tol
    verdict = "PASS" if ok else "FAIL"
    severity = "INFO" if ok else "WARNING"
    explanation = (f"Expected {expected:.4f} = {unit_price} * {quantity}; "
                   f"claimed {claimed_total} (diff {diff:.4f}, tol {tol:.4f}).")
    return {"rule_id": "R15", "verdict": verdict, "severity": severity,
            "expected": expected, "diff": diff, "tolerance": tol,
            "explanation": explanation}


def _tool_verify_date_arithmetic(asn_delivery_date: str, requested_delivery_date: str,
                                  lower_days: int, upper_days: int) -> dict:
    """R03 (Delivery Date Feasibility): asn_delivery_date within
    [requestedDeliveryDate - lower_days, requestedDeliveryDate + upper_days]."""
    ad = _parse_date_safe(asn_delivery_date)
    rd = _parse_date_safe(requested_delivery_date)
    if ad is None or rd is None:
        return {"rule_id": "R03", "verdict": "FAIL", "severity": "WARNING",
                "explanation": f"Unparseable date(s): asn={asn_delivery_date!r}, req={requested_delivery_date!r}."}
    delta_days = (ad - rd).days
    in_window = -int(lower_days or 0) <= delta_days <= int(upper_days or 0)
    verdict = "PASS" if in_window else "FAIL"
    severity = "INFO" if in_window else "WARNING"
    explanation = (f"asn_delivery {asn_delivery_date} vs requested {requested_delivery_date}: "
                   f"delta={delta_days}d, window=[-{lower_days}, +{upper_days}]d -> "
                   f"{'within' if in_window else 'outside'}.")
    return {"rule_id": "R03", "verdict": verdict, "severity": severity,
            "delta_days": delta_days, "explanation": explanation}


def _tool_verify_line_completeness(po_line_numbers: list, asn_line_numbers: list) -> dict:
    """R06 (Line Item Completeness): every PO line referenced by the ASN
    portion must have a matching ShipNoticeItem. Partial coverage downgrades
    CRITICAL -> WARNING (mirrors the deterministic R06 partial-delivery path)."""
    po_set = set(int(x) for x in (po_line_numbers or []))
    asn_set = set(int(x) for x in (asn_line_numbers or []))
    missing = sorted(po_set - asn_set)
    extra = sorted(asn_set - po_set)
    if not po_set:
        return {"rule_id": "R06", "verdict": "FAIL", "severity": "CRITICAL",
                "missing_lines": [], "extra_lines": extra,
                "explanation": "PO line set empty -- cannot evaluate completeness."}
    covered = po_set & asn_set
    if not missing:
        return {"rule_id": "R06", "verdict": "PASS", "severity": "INFO",
                "missing_lines": [], "extra_lines": extra,
                "explanation": f"All PO lines {sorted(po_set)} present in ASN."}
    if covered:
        return {"rule_id": "R06", "verdict": "FAIL", "severity": "WARNING",
                "missing_lines": missing, "extra_lines": extra,
                "explanation": (f"Partial delivery: ASN covers {sorted(covered)}, "
                                f"missing {missing}. CRITICAL downgraded -> WARNING.")}
    return {"rule_id": "R06", "verdict": "FAIL", "severity": "CRITICAL",
            "missing_lines": missing, "extra_lines": extra,
            "explanation": f"All PO lines {sorted(po_set)} missing from ASN portion."}


def _tool_verify_partial_shipment(asn_qty: float, po_qty: float,
                                   lower_pct: float = 0.0) -> dict:
    """R16 (Partial Shipment Handling): classify ASN qty against PO qty as
    in_full, partial (legitimate under-ship), or over_ship. Mirrors the R01
    three-way split that drives R16 INFO -> WARNING escalation."""
    lower = po_qty * (1 - lower_pct / 100.0)
    if asn_qty > po_qty:
        verdict, severity, status = "FAIL", "CRITICAL", "over_ship"
        explanation = f"ASN qty {asn_qty} > PO qty {po_qty}: over-ship (R01 territory)."
    elif asn_qty < lower:
        verdict, severity, status = "FAIL", "WARNING", "under_ship"
        explanation = (f"ASN qty {asn_qty} below lower bound {lower:.4g}: legitimate partial "
                       f"-> R16 escalated INFO -> WARNING.")
    elif asn_qty < po_qty:
        verdict, severity, status = "PASS", "INFO", "partial_within_tol"
        explanation = f"ASN qty {asn_qty} under PO qty {po_qty} but within lower tolerance."
    else:
        verdict, severity, status = "PASS", "INFO", "in_full"
        explanation = f"ASN qty {asn_qty} == PO qty {po_qty}: shipped in full."
    return {"rule_id": "R16", "verdict": verdict, "severity": severity,
            "status": status, "explanation": explanation}


def _tool_verify_shipment_id_length(shipment_id: str, max_length: int = 35) -> dict:
    """R19 (ShipmentID Length): Ariba SCC silently drops shipmentIDs longer
    than 35 characters."""
    n = len(shipment_id or "")
    ok = n <= int(max_length)
    verdict = "PASS" if ok else "FAIL"
    severity = "INFO" if ok else "CRITICAL"
    explanation = (f"shipmentID length {n} {'<=' if ok else '>'} {max_length}: "
                   f"{'within' if ok else 'over'} Ariba limit.")
    return {"rule_id": "R19", "verdict": verdict, "severity": severity,
            "length": n, "limit": int(max_length), "explanation": explanation}


def _parse_date_safe(s):
    """Date parser used by tool wrappers; mirrors the Stage 6 enforcer parser."""
    if not s:
        return None
    s2 = re.sub(r"[+-]\d{2}:\d{2}$", "", s).replace("Z", "")
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s2, fmt).date()
        except ValueError:
            continue
    return None


_STAGE5_TOOL_IMPLS = {
    "verify_quantity_match":     _tool_verify_quantity_match,
    "verify_price_match":        _tool_verify_price_match,
    "verify_line_total":         _tool_verify_line_total,
    "verify_date_arithmetic":    _tool_verify_date_arithmetic,
    "verify_line_completeness":  _tool_verify_line_completeness,
    "verify_partial_shipment":   _tool_verify_partial_shipment,
    "verify_shipment_id_length": _tool_verify_shipment_id_length,
}

# Tool name -> rule_id mapping for the audit log.
_TOOL_RULE_MAP = {
    "verify_quantity_match":     "R01",
    "verify_price_match":        "R02",
    "verify_date_arithmetic":    "R03",
    "verify_line_completeness":  "R06",
    "verify_line_total":         "R15",
    "verify_partial_shipment":   "R16",
    "verify_shipment_id_length": "R19",
}

# Azure OpenAI tool schemas (function-calling). Argument names mirror those of
# the Python wrappers above so the executor can pass them through directly.
_STAGE5_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "verify_quantity_match",
            "description": ("R01 Shipped Quantity Match: classify ASN qty against PO qty using the "
                            "asymmetric tolerance band [po_qty*(1-lower_pct/100), po_qty*(1+upper_pct/100)]. "
                            "Three-way result: in_range PASS, under_ship PASS (queues R16), over_ship FAIL CRITICAL."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["asn_qty", "po_qty", "lower_pct", "upper_pct"],
                "properties": {
                    "asn_qty":   {"type": "number"},
                    "po_qty":    {"type": "number"},
                    "lower_pct": {"type": "number", "description": "PO baseline tolerances.quantity_lower_pct"},
                    "upper_pct": {"type": "number", "description": "PO baseline tolerances.quantity_upper_pct"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_price_match",
            "description": ("R02 Unit Price Match: ASN unit price within asymmetric band "
                            "[po_price*(1-lower_pct/100), po_price*(1+upper_pct/100)]."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["asn_price", "po_price", "lower_pct", "upper_pct"],
                "properties": {
                    "asn_price": {"type": "number"},
                    "po_price":  {"type": "number"},
                    "lower_pct": {"type": "number", "description": "PO baseline tolerances.price_lower_pct"},
                    "upper_pct": {"type": "number", "description": "PO baseline tolerances.price_upper_pct"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_line_total",
            "description": ("R15 Total Value Reconciliation: unit_price * quantity equals claimed_total "
                            "within tolerance max(0.01, expected*0.0001)."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["unit_price", "quantity", "claimed_total"],
                "properties": {
                    "unit_price":    {"type": "number"},
                    "quantity":      {"type": "number"},
                    "claimed_total": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_date_arithmetic",
            "description": ("R03 Delivery Date Feasibility: asn_delivery_date within "
                            "[requestedDeliveryDate - lower_days, requestedDeliveryDate + upper_days]."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["asn_delivery_date", "requested_delivery_date", "lower_days", "upper_days"],
                "properties": {
                    "asn_delivery_date":       {"type": "string", "description": "ASN deliveryDate (YYYY-MM-DD or ISO-8601)"},
                    "requested_delivery_date": {"type": "string", "description": "PO requestedDeliveryDate"},
                    "lower_days":              {"type": "integer", "description": "PO baseline tolerances.time_lower_days"},
                    "upper_days":              {"type": "integer", "description": "PO baseline tolerances.time_upper_days"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_line_completeness",
            "description": ("R06 Line Item Completeness: every PO line must have a matching ShipNoticeItem "
                            "in the portion. Partial coverage downgrades CRITICAL -> WARNING."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["po_line_numbers", "asn_line_numbers"],
                "properties": {
                    "po_line_numbers":  {"type": "array", "items": {"type": "integer"}},
                    "asn_line_numbers": {"type": "array", "items": {"type": "integer"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_partial_shipment",
            "description": ("R16 Partial Shipment Handling: classify ASN qty against PO qty as "
                            "in_full / partial_within_tol / under_ship / over_ship. Under-ship escalates "
                            "R16 INFO -> WARNING (REVIEW disposition)."),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["asn_qty", "po_qty", "lower_pct"],
                "properties": {
                    "asn_qty":   {"type": "number"},
                    "po_qty":    {"type": "number"},
                    "lower_pct": {"type": "number", "description": "PO baseline tolerances.quantity_lower_pct"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_shipment_id_length",
            "description": "R19 ShipmentID Length: shipmentID must be at most max_length characters (Ariba SCC limit is 35).",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["shipment_id", "max_length"],
                "properties": {
                    "shipment_id": {"type": "string"},
                    "max_length":  {"type": "integer"},
                },
            },
        },
    },
]


# ══════════════════════════════════════════════════════════════
# Stage 1 — PO Baseline Capture
# ══════════════════════════════════════════════════════════════

def _text(el, path, default=""):
    node = el.find(path)
    return (node.text or "").strip() if node is not None else default


def _attr(el, path, attr, default=""):
    node = el.find(path)
    return node.get(attr, default) if node is not None else default


def _transport_terms(el, path=".//TransportTerms", default=""):
    """Read the incoterm code from a TransportTerms element.

    cXML carries the code in the @value attribute, but when the code is not in
    the enumerated cXML list the partner sets value="Other" and puts the real
    code in the element text, e.g. <TransportTerms value="Other">FCA</...>.
    Prefer the text in that case so we read "FCA" rather than "Other".
    """
    node = el.find(path) if el is not None else None
    if node is None:
        return default
    value = (node.get("value", "") or "").strip()
    text = (node.text or "").strip()
    if value.lower() == "other" and text:
        return text
    return value or text or default


def _parse_tolerances(item_out):
    tol = {
        "quantity_lower_pct": 0.0, "quantity_upper_pct": 0.0,
        "price_lower_pct": 0.0, "price_upper_pct": 0.0,
        "time_lower_days": 0, "time_upper_days": 0,
    }
    oc_instr = item_out.find(".//OCInstruction")
    if oc_instr is None:
        return tol
    lower = oc_instr.find("Lower/Tolerances")
    upper = oc_instr.find("Upper/Tolerances")
    if lower is not None:
        tol["quantity_lower_pct"] = _parse_money(_attr(lower, "QuantityTolerance/Percentage", "percent", "0"))
        tol["price_lower_pct"] = _parse_money(_attr(lower, "PriceTolerance/Percentage", "percent", "0"))
        tol["time_lower_days"] = int(_attr(lower, "TimeTolerance", "limit", "0"))
    if upper is not None:
        tol["quantity_upper_pct"] = _parse_money(_attr(upper, "QuantityTolerance/Percentage", "percent", "0"))
        tol["price_upper_pct"] = _parse_money(_attr(upper, "PriceTolerance/Percentage", "percent", "0"))
        tol["time_upper_days"] = int(_attr(upper, "TimeTolerance", "limit", "0"))
    return tol


def _parse_address(addr_el):
    if addr_el is None:
        return {}
    streets = [s.text for s in addr_el.findall(".//Street") if s.text]
    return {
        "addressID": addr_el.get("addressID", ""),
        "name": _text(addr_el, "Name"),
        "street": ", ".join(streets),
        "city": _text(addr_el, ".//City"),
        "postalCode": _text(addr_el, ".//PostalCode"),
        "country": _attr(addr_el, ".//Country", "isoCountryCode"),
    }


def parse_po_cxml(filepath) -> dict:
    tree = etree.parse(str(filepath))
    root = tree.getroot()
    header = root.find(".//OrderRequestHeader")

    baseline = {
        "orderID": header.get("orderID"),
        "orderDate": header.get("orderDate"),
        "orderType": header.get("orderType", "regular"),
        "orderVersion": header.get("orderVersion", "1"),
        "currency": _attr(header, "Total/Money", "currency"),
        "total": _parse_money(_text(header, "Total/Money", "0")),
        "shipTo": _parse_address(header.find(".//ShipTo/Address")),
        "billTo": _parse_address(header.find(".//BillTo/Address")),
        "paymentTermDays": int(header.find(".//PaymentTerm").get("payInNumberOfDays", "0"))
            if header.find(".//PaymentTerm") is not None else 0,
        "incoterms": _transport_terms(header),
        "incotermsLocation": _text(header, ".//TermsOfDelivery/Address/Name"),
    }

    supplier_contact = root.find(".//Contact[@role='supplierCorporate']")
    from_cred = root.find(".//To/Credential[@domain='NetworkID']/Identity")
    baseline["supplier"] = {
        "networkID": from_cred.text.strip() if from_cred is not None else "",
        "name": _text(supplier_contact, "Name") if supplier_contact is not None else "",
        "buyerID": supplier_contact.get("addressID", "") if supplier_contact is not None else "",
    }

    baseline["lineItems"] = []
    for item in root.findall(".//ItemOut"):
        line_num = item.get("lineNumber", "")
        line = {
            "lineNumber": int(line_num),
            "lineNumberRaw": line_num,
            "quantity": _parse_money(item.get("quantity", "0")),
            "requestedDeliveryDate": item.get("requestedDeliveryDate", ""),
            "requestedShipmentDate": item.get("requestedShipmentDate", ""),
            "supplierPartID": _text(item, "ItemID/SupplierPartID"),
            "buyerPartID": _text(item, "ItemID/BuyerPartID"),
            "unitPrice": _parse_money(_text(item, "ItemDetail/UnitPrice/Money", "0")),
            "currency": _attr(item, "ItemDetail/UnitPrice/Money", "currency"),
            "description": _text(item, "ItemDetail/Description"),
            "unitOfMeasure": _text(item, "ItemDetail/UnitOfMeasure"),
            "tolerances": _parse_tolerances(item),
        }
        baseline["lineItems"].append(line)

    return baseline


# ══════════════════════════════════════════════════════════════
# Stage 2 — ASN Ingestion
# ══════════════════════════════════════════════════════════════

def _parse_contact(contact_el):
    if contact_el is None:
        return {}
    streets = [s.text for s in contact_el.findall(".//Street") if s.text]
    return {
        "addressID": contact_el.get("addressID", ""),
        "role": contact_el.get("role", ""),
        "name": _text(contact_el, "Name"),
        "street": ", ".join(streets),
        "city": _text(contact_el, ".//City"),
        "postalCode": _text(contact_el, ".//PostalCode"),
        "country": _attr(contact_el, ".//Country", "isoCountryCode"),
    }


def ingest_asn_from_xml(raw_xml: str) -> dict:
    """Ingest ASN from raw XML string (used by API)."""
    tree = etree.fromstring(raw_xml.encode("utf-8"))
    ship_header = tree.find(".//ShipNoticeHeader")

    supplier_network_id = ""
    from_cred = tree.find(".//From/Credential[@domain='NetworkID']/Identity")
    if from_cred is not None:
        supplier_network_id = from_cred.text.strip()

    vendor_id = ""
    vendor_cred = tree.find(".//From/Credential[@domain='VendorID']/Identity")
    if vendor_cred is not None:
        vendor_id = vendor_cred.text.strip()

    referenced_pos = []
    for portion in tree.findall(".//ShipNoticePortion"):
        order_ref = portion.find("OrderReference")
        if order_ref is not None:
            po_id = order_ref.get("orderID", "")
            if po_id and po_id not in referenced_pos:
                referenced_pos.append(po_id)

    packaging = {}
    pkg_el = ship_header.find("Packaging") if ship_header is not None else None
    if pkg_el is not None:
        for dim in pkg_el.findall("Dimension"):
            dim_type = dim.get("type", "")
            dim_qty = dim.get("quantity", "0")
            dim_uom = _text(dim, "UnitOfMeasure")
            packaging[dim_type] = {"quantity": _parse_money(dim_qty), "uom": dim_uom}

    return {
        "raw_xml": raw_xml,
        "shipmentID": ship_header.get("shipmentID", "") if ship_header is not None else "",
        "shipmentDate": ship_header.get("shipmentDate", "") if ship_header is not None else "",
        "deliveryDate": ship_header.get("deliveryDate", "") if ship_header is not None else "",
        "noticeDate": ship_header.get("noticeDate", "") if ship_header is not None else "",
        "operation": ship_header.get("operation", "") if ship_header is not None else "",
        "referencedPOs": referenced_pos,
        "shipFrom": _parse_contact(ship_header.find("Contact[@role='shipFrom']") if ship_header is not None else None),
        "shipTo": _parse_contact(ship_header.find("Contact[@role='shipTo']") if ship_header is not None else None),
        "transportTerms": _transport_terms(ship_header) if ship_header is not None else "",
        "packaging": packaging,
        "payloadID": tree.get("payloadID", ""),
        "supplierNetworkID": supplier_network_id,
        "supplierVendorID": vendor_id,
        "status": "PENDING",
        "ingestedAt": datetime.now(timezone.utc).isoformat(),
    }


def ingest_asn(filepath) -> dict:
    """Read a raw ASN cXML file and extract tracking metadata."""
    raw_xml = pathlib.Path(filepath).read_text(encoding="utf-8")
    return ingest_asn_from_xml(raw_xml)


# ══════════════════════════════════════════════════════════════
# Stage 3 — LLM Parsing & Normalization
# ══════════════════════════════════════════════════════════════

ASN_PARSE_SYSTEM_PROMPT = """You are an expert cXML Advanced Shipping Notice (ASN) parser.
Given raw cXML for a ShipNoticeRequest, extract a normalized JSON object
with the following structure. Output ONLY valid JSON, no extra text.

{
  "shipmentID": "<from ShipNoticeHeader>",
  "shipmentDate": "<ISO datetime>",
  "deliveryDate": "<ISO datetime>",
  "noticeDate": "<ISO datetime>",
  "operation": "<new|update|delete>",
  "shipFrom": {
    "name": "<string>",
    "street": "<string>",
    "city": "<string>",
    "postalCode": "<string>",
    "country": "<ISO country code>"
  },
  "shipTo": {
    "addressID": "<string>",
    "name": "<string>",
    "street": "<string>",
    "city": "<string>",
    "postalCode": "<string>",
    "country": "<ISO country code>"
  },
  "transportTerms": "<e.g. FCA>",
  "packaging": {
    "grossVolume": {"quantity": <float>, "uom": "<string>"},
    "grossWeight": {"quantity": <float>, "uom": "<string>"},
    "netWeight": {"quantity": <float>, "uom": "<string>"}
  },
  "portions": [
    {
      "referencedPO": "<orderID from OrderReference>",
      "referencedPODate": "<orderDate from OrderReference>",
      "items": [
        {
          "shipNoticeLineNumber": <int>,
          "lineNumber": <int, normalized from PO line — e.g. "00010" -> 10>,
          "quantity": <float>,
          "supplierPartID": "<string>",
          "buyerPartID": "<string>",
          "unitPrice": <float>,
          "currency": "<string>",
          "description": "<string>",
          "unitOfMeasure": "<string>",
          "unitNetWeight": <float, line-level net weight from ShipNoticeItemDetail/Dimension type="unitNetWeight"; 0 if absent>
        }
      ]
    }
  ]
}

Rules:
- lineNumber must be normalized to integer (e.g., "00010" -> 10)
- shipNoticeLineNumber is the ASN-level sequence number
- quantity and unitPrice must be numbers
- Extract ALL ShipNoticePortion and ShipNoticeItem elements
- Group items by their ShipNoticePortion (each portion references one PO)
- packaging: extract every Packaging/Dimension. Include netWeight whenever a
  Dimension type="netWeight" is present (omit only if truly absent) — it is
  required for the weight-plausibility check (R20)
- unitNetWeight: per item, extract ShipNoticeItemDetail/Dimension
  type="unitNetWeight" (the line-level net weight); use 0 if absent. Net weight
  is often provided here at the line level rather than in Packaging.netWeight
- If a field is missing, use empty string for strings, 0 for numbers
"""

ASN_PARSE_FEW_SHOT = """Example input (abbreviated):
<ShipNoticeHeader shipmentID="12345" shipmentDate="2025-11-04T03:00:00" deliveryDate="2025-11-06T03:00:00" noticeDate="2025-11-03T01:55:04" operation="new">
  <Contact role="shipFrom"><Name>Supplier Co</Name>...</Contact>
  <Contact role="shipTo"><Name>Hilti Warehouse</Name>...</Contact>
  <TransportTerms value="Other">FCA</TransportTerms>
</ShipNoticeHeader>
<ShipNoticePortion>
  <OrderReference orderID="ORDER_SAMPLE_01" orderDate="2025-10-15T05:00:00"/>
  <ShipNoticeItem shipNoticeLineNumber="1" lineNumber="10" quantity="100.000">
    <ItemID><SupplierPartID>ABC123</SupplierPartID><BuyerPartID>BUYER_PART_SAMPLE</BuyerPartID></ItemID>
    <ShipNoticeItemDetail>
      <UnitPrice><Money currency="EUR">5.50</Money></UnitPrice>
      <Description>Widget A</Description>
      <UnitOfMeasure>H87</UnitOfMeasure>
    </ShipNoticeItemDetail>
  </ShipNoticeItem>
</ShipNoticePortion>

Example output:
{"shipmentID":"12345","shipmentDate":"2025-11-04T03:00:00","deliveryDate":"2025-11-06T03:00:00","noticeDate":"2025-11-03T01:55:04","operation":"new","shipFrom":{"name":"Supplier Co","street":"","city":"","postalCode":"","country":""},"shipTo":{"addressID":"","name":"Hilti Warehouse","street":"","city":"","postalCode":"","country":""},"transportTerms":"FCA","packaging":{"grossVolume":{"quantity":0,"uom":""},"grossWeight":{"quantity":0,"uom":""},"netWeight":{"quantity":0,"uom":""}},"portions":[{"referencedPO":"ORDER_SAMPLE_01","referencedPODate":"2025-10-15T05:00:00","items":[{"shipNoticeLineNumber":1,"lineNumber":10,"quantity":100.0,"supplierPartID":"ABC123","buyerPartID":"BUYER_PART_SAMPLE","unitPrice":5.5,"currency":"EUR","description":"Widget A","unitOfMeasure":"H87","unitNetWeight":0}]}]}
"""


# Stage 2 structured-outputs schema. Fields mirror ASN_PARSE_SYSTEM_PROMPT.
# Strict mode constraints: every property listed in `required`,
# `additionalProperties: false`, no `oneOf`/`pattern`/numeric bounds.
_ADDRESS_PROPS_BASE = {
    "name":       {"type": "string"},
    "street":     {"type": "string"},
    "city":       {"type": "string"},
    "postalCode": {"type": "string"},
    "country":    {"type": "string"},
}
_ASN_PARSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["shipmentID", "shipmentDate", "deliveryDate", "noticeDate",
                 "operation", "shipFrom", "shipTo", "transportTerms",
                 "packaging", "portions"],
    "properties": {
        "shipmentID":     {"type": "string"},
        "shipmentDate":   {"type": "string"},
        "deliveryDate":   {"type": "string"},
        "noticeDate":     {"type": "string"},
        "operation":      {"type": "string"},
        "transportTerms": {"type": "string"},
        "shipFrom": {
            "type": "object",
            "additionalProperties": False,
            "required": list(_ADDRESS_PROPS_BASE.keys()),
            "properties": _ADDRESS_PROPS_BASE,
        },
        "shipTo": {
            "type": "object",
            "additionalProperties": False,
            "required": ["addressID"] + list(_ADDRESS_PROPS_BASE.keys()),
            "properties": {"addressID": {"type": "string"}, **_ADDRESS_PROPS_BASE},
        },
        "packaging": {
            "type": "object",
            "additionalProperties": False,
            # netWeight is required by the strict structured-output schema (all
            # properties must appear in `required`). When the ASN has no
            # netWeight, emit {"quantity": 0, "uom": ""} — the same
            # absent-sentinel convention grossVolume/grossWeight use. R20
            # treats that sentinel as "absent" (R21's territory).
            "required": ["grossVolume", "grossWeight", "netWeight"],
            "properties": {
                "grossVolume": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["quantity", "uom"],
                    "properties": {
                        "quantity": {"type": "number"},
                        "uom":      {"type": "string"},
                    },
                },
                "grossWeight": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["quantity", "uom"],
                    "properties": {
                        "quantity": {"type": "number"},
                        "uom":      {"type": "string"},
                    },
                },
                # netWeight is optional: many ASNs omit it (R21 owns the
                # absence check), but when present it MUST reach the validator
                # so R20's gross/net ratio check has an input. Without this
                # property `additionalProperties: False` silently drops it.
                "netWeight": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["quantity", "uom"],
                    "properties": {
                        "quantity": {"type": "number"},
                        "uom":      {"type": "string"},
                    },
                },
            },
        },
        "portions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["referencedPO", "referencedPODate", "items"],
                "properties": {
                    "referencedPO":     {"type": "string"},
                    "referencedPODate": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["shipNoticeLineNumber", "lineNumber",
                                         "quantity", "supplierPartID",
                                         "buyerPartID", "unitPrice", "currency",
                                         "description", "unitOfMeasure",
                                         "unitNetWeight"],
                            "properties": {
                                "shipNoticeLineNumber": {"type": "integer"},
                                "lineNumber":           {"type": "integer"},
                                "quantity":             {"type": "number"},
                                "supplierPartID":       {"type": "string"},
                                "buyerPartID":          {"type": "string"},
                                "unitPrice":            {"type": "number"},
                                "currency":             {"type": "string"},
                                "description":          {"type": "string"},
                                "unitOfMeasure":        {"type": "string"},
                                # Line-level net weight (ShipNoticeItemDetail/
                                # Dimension type="unitNetWeight"); 0 if absent.
                                # Lets R20/R21 recognise net weight provided at
                                # the line level rather than in Packaging.
                                "unitNetWeight":        {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
    },
}


def llm_parse_asn(asn_raw_xml: str) -> dict:
    user_prompt = (
        f"{ASN_PARSE_FEW_SHOT}\n\n"
        f"Now parse this Advanced Shipping Notice cXML:\n\n{asn_raw_xml}"
    )
    schema = _ASN_PARSE_SCHEMA if USE_STAGE2_SCHEMA else None
    raw_response = llm_call(ASN_PARSE_SYSTEM_PROMPT, user_prompt,
                            schema=schema, schema_name="asn_parse")
    # Try parsing, with fallback for control characters
    for text in [raw_response, re.sub(r'[\x00-\x1f\x7f]', ' ', raw_response)]:
        try:
            parsed = json.loads(text)
            if "error" in parsed and not parsed.get("portions"):
                print(f"  [WARNING] LLM call failed, returning empty parse with error flag")
                parsed["_llm_error"] = True
            return parsed
        except json.JSONDecodeError:
            continue
    # All attempts failed
    print(f"  [ERROR] LLM returned unparseable JSON in Stage 3")
    print(f"  Raw response (first 500 chars): {raw_response[:500]}")
    return {
        "shipmentID": "unknown",
        "portions": [],
        "overallAssessment": "LLM parse returned invalid JSON",
        "_llm_error": True,
    }


# ══════════════════════════════════════════════════════════════
# Stage 4 — Prompt Builder & Context Integrator
# ══════════════════════════════════════════════════════════════

RULE_DEFINITIONS = [
    {"id": "R01", "name": "Shipped Quantity Match",       "severity": "CRITICAL", "description": "ASN shipped quantity must be within PO tolerance band (quantity_lower_pct / quantity_upper_pct from ControlKeys)."},
    {"id": "R02", "name": "Unit Price Match",             "severity": "CRITICAL", "description": "ASN unit price must match PO unit price within tolerance band (price_lower_pct / price_upper_pct)."},
    {"id": "R03", "name": "Delivery Date Feasibility",    "severity": "WARNING",  "description": "ASN delivery date must fall within PO tolerance window (time_lower_days / time_upper_days from requestedDeliveryDate)."},
    {"id": "R04", "name": "Shipment Date Realism",        "severity": "WARNING",  "description": "ASN shipmentDate must not be in the past relative to noticeDate, and deliveryDate must be >= shipmentDate."},
    {"id": "R05", "name": "UOM Consistency",              "severity": "CRITICAL", "description": "ASN unit of measure must match PO unit of measure under canonical-class equivalence (UN/CEFACT synonyms count as equal). Examples: PCE ≡ H87 ≡ EA ≡ PC (all 'piece'); KGM ≡ KG (kilogram); MTR ≡ M (metre). FAIL only when canonical classes differ (e.g. PCE vs KGM)."},
    {"id": "R06", "name": "Line Item Completeness",       "severity": "CRITICAL", "description": "Every PO line referenced in the ASN portion must have a corresponding ASN ShipNoticeItem."},
    {"id": "R07", "name": "No Phantom Lines",             "severity": "CRITICAL", "description": "ASN must not contain items with lineNumbers that have no corresponding PO line."},
    {"id": "R08", "name": "Currency Match",               "severity": "CRITICAL", "description": "ASN currency must match PO currency. Comparison rule: if the ASN line-level <Money currency='...'> attribute is empty or absent, fall back to the ASN header-level currency before declaring a mismatch (cXML allows currency-at-header with line-level inheritance). FAIL only when both ASN line currency AND header currency disagree with PO currency."},
    {"id": "R09", "name": "Supplier ID Validation",       "severity": "CRITICAL", "description": "ASN supplier credential must match PO supplier credential at the comparable level. The canonical check is ASN-NetworkID against PO-NetworkID — Ariba publishes NetworkID on every PO and the supplier should echo it on the ASN. When the ASN carries only VendorID (the buyer-internal ANID), no arithmetic comparison is possible because PO does not publish VendorID; identity is then trusted from the upstream PO-selection step. R09 fires only when (a) ASN+PO both carry NetworkID and they differ, or (b) the ASN carries no supplier credential at all."},
    {"id": "R10", "name": "PO Reference Integrity",       "severity": "CRITICAL", "description": "Each ASN ShipNoticePortion must reference a valid, existing PO orderID."},
    {"id": "R11", "name": "Schema Compliance",            "severity": "INFO",     "description": "ASN must be valid cXML ShipNoticeRequest that parsed without errors."},
    {"id": "R12", "name": "Mandatory Field Presence",     "severity": "WARNING",  "description": "ASN must have: shipmentID, shipmentDate, deliveryDate, and each item must have lineNumber, quantity, unitOfMeasure."},
    {"id": "R13", "name": "Transport Terms Match",        "severity": "WARNING",  "description": "ASN transport terms (e.g., FCA) must match PO incoterms."},
    {"id": "R14", "name": "Ship-To Address Match",        "severity": "WARNING",  "description": "ASN shipTo address (addressID or name+country) must match PO shipTo address."},
    {"id": "R15", "name": "Total Value Reconciliation",   "severity": "WARNING",  "description": "ASN line total (shipped qty * unitPrice) must match PO line total within rounding tolerance."},
    {"id": "R16", "name": "Partial Shipment Handling",    "severity": "INFO",     "description": "If ASN ships less than PO quantity, flag as partial shipment with remaining quantity."},
    {"id": "R17", "name": "Multi-PO Consolidation Check", "severity": "WARNING",  "description": "For multi-PO ASNs, verify each ShipNoticePortion correctly maps to its referenced PO and items don't cross PO boundaries."},
    {"id": "R18", "name": "Duplicate ASN Detection",      "severity": "WARNING",  "description": "Same shipmentID must not appear more than once for the same set of POs. Treated as WARNING because a stateful duplicate flag alone is not enough evidence to block goods receipt; a buyer should verify before rejecting."},
    {"id": "R19", "name": "ShipmentID Length",            "severity": "CRITICAL", "description": "shipmentID must be 35 characters or fewer. Ariba SCC cannot process ASN names longer than 35 characters and will silently drop the document."},
    {"id": "R20", "name": "Weight Plausibility",          "severity": "WARNING",  "description": "If grossWeight and netWeight are present in Packaging: (a) grossWeight must be STRICTLY > netWeight (Buyer Guide §4.5/§4.6), and (b) grossWeight must be <= netWeight * 10 (Buyer Guide §4.2). Extreme ratio suggests data entry error."},
    {"id": "R21", "name": "Mandatory Packaging Fields",   "severity": "WARNING",   "description": "Packaging section must include: netWeight with a valid mass UoM (KGM/GRM/LBR/...), grossWeight with a valid mass UoM, and at least one packaging type (CARTON / EUPAL / CASE / packing slip). Net weight may instead be provided at the line level as item unitNetWeight; when any item carries a positive unitNetWeight, treat net weight as PRESENT (its placement in the line rather than Packaging is at most an R11 schema-placement note, NOT a completeness failure) and do not flag netWeight missing. When grossVolume is present its UoM must be a volume unit (MTQ/LTR/...). Severity is WARNING (not CRITICAL): real-world Hilti buyer practice accepts ASNs with partial packaging on a manual-review basis rather than blocking goods receipt. Use R20 to flag implausible weights when both are present (Buyer Guide §4.7, §4.8, §5.2, §7.1, §7.2)."},
    {"id": "R22", "name": "Order Fulfillment Status",     "severity": "WARNING",  "description": "ASN should not be submitted for a PO line that is already fully shipped (advised quantity equals or exceeds PO quantity when considering prior ASNs). Flag if total notified quantity exceeds PO quantity."},
]

# rule_id -> registered severity (CRITICAL / WARNING / INFO).  Used by Stage 6
# (every enforcer mode) to clamp LLM-emitted severities to the value defined
# in the rule register.
_RULE_SEVERITY = {r["id"]: r["severity"] for r in RULE_DEFINITIONS}


def _cap_rule_severities(rule_results: list, enforcer_log: list,
                          scope: tuple | None = None,
                          po_baselines: dict | None = None,
                          asn_parsed: dict | None = None) -> int:
    """Assign each entry's severity authoritatively (severity-assignment step).

    For most rules the authoritative severity is the static registered value
    (Table A.1) and this clamps the LLM's emission back to it.  A few rules
    have a *data-dependent* severity that the static register cannot express:

      - R06 (Line Item Completeness) is CRITICAL on a full miss but only
        WARNING on a legitimate partial delivery.  Clamping a fired R06 to its
        registered CRITICAL over-rejects partial deliveries (Section 5.4.2).

    When `po_baselines` and `asn_parsed` are supplied we source those
    conditional severities from the same `_classify_R06` the deterministic and
    partition enforcers use, so severity-ASSIGNMENT authority is consistent
    across modes.  This deliberately does NOT touch rule-FIRING: a rule the
    mode fired stays fired; only its severity is corrected.

    Mutates `rule_results` in place; appends one RULE_SEVERITY_CAPPED entry
    per cap to `enforcer_log`.  When `scope` is provided, only entries whose
    rule_id is in `scope` are considered (used by partition mode to restrict
    capping to LLM-emitted non-owned rules).  Returns the number of caps
    applied for callers that want to surface the count in `enforcer_summary`.
    """
    # Pre-compute data-dependent severities for the rules that have them.
    conditional: dict[str, str] = {}
    if po_baselines is not None and asn_parsed is not None:
        r06 = _classify_R06(po_baselines, asn_parsed)
        if r06.get("status") == "FAIL" and r06.get("severity"):
            conditional["R06"] = r06["severity"]  # CRITICAL (full) or WARNING (partial)

    n_caps = 0
    for r in rule_results or []:
        rid = r.get("rule_id")
        if scope is not None and rid not in scope:
            continue
        expected = _RULE_SEVERITY.get(rid)
        # A fired conditional-severity rule takes the data-dependent severity.
        if rid in conditional and r.get("status") == "FAIL":
            expected = conditional[rid]
        if expected and r.get("severity") != expected:
            llm_said = r.get("severity")
            enforcer_log.append({
                "type": "RULE_SEVERITY_CAPPED",
                "rule_id": rid,
                "llm_said": llm_said,
                "capped_to": expected,
                **({"conditional": True} if rid in conditional else {}),
            })
            # Persist a bracket marker in the rule detail so the cap survives in
            # the saved artefact (the structured enforcer_log is in-memory only).
            # build_enforcer_views / compute_enforcer_audit parse this marker;
            # the from->to pair lets the capping disposition counterfactual
            # reconstruct the pre-cap severity offline. Format matches the
            # existing CAPPED regex: [ENFORCER: capped <from> -> <to>].
            marker = f" [ENFORCER: capped {llm_said} -> {expected}]"
            if marker not in (r.get("detail") or ""):
                r["detail"] = (r.get("detail") or "") + marker
            r["severity"] = expected
            n_caps += 1
    return n_caps


# Stage 5 worked-example block (few-shot). Three compact cases covering the
# main failure-mode shapes: clean PASS, R01 over-shipment CRITICAL,
# R19 shipmentID-too-long CRITICAL. Calibrates the model on the exact
# `ruleResults` shape and on what counts as PASS vs FAIL.
VALIDATION_FEW_SHOT = """Below are calibration examples showing the expected ruleResults shape and
decision boundaries for each of the 22 rules. Use them as a guide -- your
output must follow the same JSON structure.

PART A. Three full worked examples
==================================

--- EXAMPLE 1: clean ASN (all rules PASS) ---
INPUT (abbreviated):
  PO ORDER_SAMPLE_01: line 10, qty 100, unitPrice 5.50 EUR, UOM H87, tol +/-5%
  ASN ship 12345: portion->ORDER_SAMPLE_01, item line 10, qty 100, unitPrice 5.50 EUR, UOM H87
EXPECTED ruleResults excerpt:
  [
    {"rule_id":"R01","status":"PASS","severity":"CRITICAL","referencedPO":"ORDER_SAMPLE_01","lineNumber":10,
     "detail":"ASN qty 100.0 within PO band [95.0, 105.0]"},
    {"rule_id":"R02","status":"PASS","severity":"CRITICAL","referencedPO":"ORDER_SAMPLE_01","lineNumber":10,
     "detail":"ASN unitPrice 5.50 == PO 5.50"},
    {"rule_id":"R19","status":"PASS","severity":"CRITICAL","referencedPO":null,"lineNumber":null,
     "detail":"shipmentID '12345' length 5 <= 35"}
  ]

--- EXAMPLE 2: R01 over-shipment (CRITICAL FAIL) ---
INPUT (abbreviated):
  PO ORDER_SAMPLE_02: line 10, qty 100, tol +/-5%
  ASN ship 67890: portion->ORDER_SAMPLE_02, item line 10, qty 130
EXPECTED ruleResults excerpt:
  [
    {"rule_id":"R01","status":"FAIL","severity":"CRITICAL","referencedPO":"ORDER_SAMPLE_02","lineNumber":10,
     "detail":"ASN qty 130.0 exceeds PO upper bound 105.0 (over-ship by 25.0 = 25%)"}
  ]
NOTE: Do NOT also flag R22 unless prior ASNs are referenced. One defect, one rule fail.

--- EXAMPLE 3: R19 shipmentID length (CRITICAL FAIL) ---
INPUT (abbreviated):
  PO ORDER_SAMPLE_03 (any line)
  ASN ship "SHIPMENT_2025_11_04_SAMPLE_SITE_REF_999_LINE_10": length 47
EXPECTED ruleResults excerpt:
  [
    {"rule_id":"R19","status":"FAIL","severity":"CRITICAL","referencedPO":null,"lineNumber":null,
     "detail":"shipmentID length 47 > 35 (Ariba SCC will silently drop the document)"}
  ]
NOTE: R19 is ASN-header level -- referencedPO and lineNumber MUST be null.

PART B. Per-rule PASS / FAIL micro-anchors
==========================================
One PASS line and one FAIL line per rule. These show the canonical decision
boundary so you do not have to extrapolate from descriptions alone. Severity
is always taken from the rule register (do not promote/demote).

R01 (CRITICAL, line-level) Shipped Quantity Match
  PASS: ASN qty 2016 vs PO qty 2016, tol +/-0% -> within band [2016, 2016]
  FAIL: ASN qty 3000 vs PO qty 2016, tol +/-0% -> exceeds upper bound 2016 (over by 49%)

R02 (CRITICAL, line-level) Unit Price Match
  PASS: ASN price 7.20 EUR vs PO price 7.20 EUR, tol +/-0% -> equal
  FAIL: ASN price 9.50 EUR vs PO price 7.20 EUR, tol +/-0% -> +31.9% over PO upper bound

R03 (WARNING, line-level) Delivery Date Feasibility
  PASS: ASN deliveryDate 2026-04-29, PO requested 2026-04-30, tol -3..+3 days -> within window
  FAIL: ASN deliveryDate 2026-05-15, PO requested 2026-04-30, tol -3..+3 days -> +15 days outside window

R04 (WARNING, header) Shipment Date Realism
  PASS: noticeDate <= shipmentDate <= deliveryDate (e.g. 2026-04-28 <= 2026-04-29 <= 2026-05-02)
  FAIL: deliveryDate 2026-04-25 < shipmentDate 2026-04-29 -> chronologically impossible

R05 (CRITICAL, line-level) UOM Consistency
  Canonical equivalence applies — UN/CEFACT synonyms count as equal:
    PIECE class:    PCE ≡ H87 ≡ EA ≡ PC ≡ C62
    MASS class:     KGM ≡ KG ; GRM ≡ G ; LBR ≡ LB ; TNE ≡ T
    LENGTH class:   MTR ≡ M ; CMT ≡ CM ; MMT ≡ MM ; INH ≡ IN ; FOT ≡ FT
    VOLUME class:   LTR ≡ L ; MLT ≡ ML
  PASS: ASN UOM 'PCE' == PO UOM 'PCE'
  PASS: ASN UOM 'PCE' vs PO UOM 'H87'  -> SAME canonical class (PIECE), accept
  PASS: ASN UOM 'KG'  vs PO UOM 'KGM'  -> SAME canonical class (KG), accept
  FAIL: ASN UOM 'PCE' vs PO UOM 'KGM'  -> different canonical class (PIECE vs KG) -> reject

R06 (CRITICAL, header/portion-level) Line Item Completeness
  PASS: Portion->PO has lines [10,20,30]; ASN items reference [10,20,30] -> complete
  FAIL: Portion->PO has lines [10,20,30]; ASN items reference only [10,20] -> line 30 missing

R07 (CRITICAL, header/portion-level) No Phantom Lines
  PASS: Every ASN item lineNumber appears in the referenced PO
  FAIL: ASN line 999 referenced; PO has no line 999 -> phantom

R08 (CRITICAL, line-level) Currency Match
  Header-fallback rule: cXML allows currency at the header with line-level
  inheritance. When the ASN line <Money currency='...'> is EMPTY or ABSENT,
  fall back to the ASN header-level currency before comparing with PO.
  PASS: ASN currency 'EUR' == PO currency 'EUR' at header AND line
  PASS: ASN line currency '' (empty), ASN header currency 'EUR' == PO 'EUR' -> accept
  FAIL: ASN line currency 'USD' vs PO 'EUR' -> CRITICAL (do not silently convert)
  FAIL: ASN line currency '' AND ASN header currency 'USD' vs PO 'EUR' -> CRITICAL

R09 (CRITICAL, header) Supplier ID Validation
  Compare same-credential-type only:
    - PRIMARY:  ASN <Credential domain="NetworkID">  vs  PO supplier networkID
    - FALLBACK: ASN <Credential domain="VendorID">   is the buyer's internal ANID;
                PO does NOT publish VendorID, so a VendorID-only ASN cannot be
                arithmetically validated against the PO. Trust upstream PO selection.
  PASS: ASN NetworkID 'AN_SYN_0001' == PO NetworkID 'AN_SYN_0001'
  PASS: ASN has no NetworkID, only VendorID 'VENDOR_SYN_01' -> not comparable, treat as PASS
        (PO selection vetted the supplier upstream)
  FAIL: ASN NetworkID 'AN_SYN_0002' != PO NetworkID 'AN_SYN_0003' -> genuine mismatch
  FAIL: ASN has neither NetworkID nor VendorID -> supplier unidentified
  DO NOT compare cross-credential types: ASN VendorID 'VENDOR_SYN_01' vs PO NetworkID
  'AN_SYN_0004' is a category error -- they are different ID schemes, not equal even
  for the same supplier.

R10 (CRITICAL, header/portion-level) PO Reference Integrity
  PASS: ASN orderID 'ORDER_ID_0008' resolves to a known PO baseline
  FAIL: ASN orderID 'ORDER_ID_9999' has no matching PO baseline -> reference broken

R11 (INFO, header) Schema Compliance
  PASS: cXML parses cleanly with no validation errors
  FAIL: cXML missing root <ShipNoticeRequest> or malformed -> schema break (rare; INFO severity)

R12 (WARNING, header/line-level) Mandatory Field Presence
  PASS: shipmentID, shipmentDate, deliveryDate present at header; lineNumber + quantity + UOM present per item
  FAIL: <Packaging> block is present but <NetWeight> element is empty -> missing mandatory subfield

R13 (WARNING, header) Transport Terms Match
  PASS: ASN transport terms 'FCA' match PO incoterms 'FCA'
  FAIL: ASN transport 'EXW' vs PO incoterms 'FCA' -> mismatch (different liability point)

R14 (WARNING, header) Ship-To Address Match
  PASS: ASN shipTo addressID matches PO shipTo addressID (or name+country if addressID absent)
  FAIL: ASN shipTo 'CITY_0024 / TOWN_A' vs PO shipTo 'CITY_0007 / TOWN_B' -> address divergent

R15 (WARNING, line-level) Total Value Reconciliation
  PASS: ASN line total (qty * unitPrice = 2016 * 7.20 = 14515.20) matches claimed line total within rounding
  FAIL: ASN line total claimed 21600.00 but qty*price = 3000*7.20 = 21600 yet PO total was 14515.20 -> divergent (collateral with R01)

R16 (INFO, line-level) Partial Shipment Handling
  PASS: ASN qty 2016 == PO qty 2016 -> not partial (or 'PASS with partial=false')
  FAIL: ASN qty 1500 < PO qty 2016 -> partial shipment, remaining 516 (INFO not blocking)

R17 (WARNING, header) Multi-PO Consolidation Check
  PASS: ASN has 2 portions, each with its own PO; items in portion A reference portion-A PO only
  FAIL: ASN portion->PO_A contains an item whose lineNumber belongs to PO_B -> cross-PO leakage

R18 (WARNING, header) Duplicate ASN Detection
  PASS: shipmentID 'SHIPMENT_ID_S03' has not been seen before for the same PO set
  FAIL: shipmentID 'SHIPMENT_ID_0031' previously seen for ORDER_ID_0068 -> duplicate (WARNING -> REVIEW)

R19 (CRITICAL, header) ShipmentID Length
  PASS: shipmentID 'SHIPMENT_ID_S07' length 15 <= 35
  FAIL: shipmentID 'SHIPMENT_ID_S19_PADDED_TO_EXCEED_35_CHARS' length 42 > 35

R20 (WARNING, line-level) Weight Plausibility
  PASS: grossWeight 12.5 kg, netWeight 10.0 kg -> ratio 1.25 within (1, 10]
  FAIL: grossWeight 250 kg, netWeight 10 kg -> ratio 25 > 10 (data entry error suspected)

R21 (WARNING, header) Mandatory Packaging Fields
  Severity is WARNING, not CRITICAL: real-world Hilti buyer practice accepts ASNs
  with partial packaging (e.g. grossWeight present but netWeight missing) on
  manual review rather than blocking goods receipt.
  PASS: <Packaging> contains netWeight (with unit), grossWeight (with unit), and a packaging type
  FAIL: <Packaging> is absent OR has no <NetWeight> AND no <GrossWeight> -> WARNING (manual review)

R22 (WARNING, line-level) Order Fulfillment Status
  PASS: cumulativeShippedQty (this ASN + prior ASNs) <= PO qty
  FAIL: cumulativeShippedQty 2300 > PO qty 2016 -> over-fulfillment (collateral with R01)

End of examples. Now evaluate the case below using these calibration anchors.
Reminder: WARNING-only failures should still produce REVIEW (not REJECT) at
disposition time. Only CRITICAL fails justify REJECT.
"""


# Stage 5 structured-outputs schema. Mirrors the JSON contract documented in
# the system prompt. Strict mode requires every property in `required` and
# nullables expressed as ["type", "null"] unions.
_RULE_RESULT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["rule_id", "name", "status", "severity", "referencedPO", "lineNumber", "detail"],
    "properties": {
        "rule_id":      {"type": "string"},
        "name":         {"type": "string"},
        "status":       {"type": "string", "enum": ["PASS", "FAIL"]},
        "severity":     {"type": "string", "enum": ["CRITICAL", "WARNING", "INFO"]},
        "referencedPO": {"type": ["string", "null"]},
        "lineNumber":   {"type": ["integer", "null"]},
        "detail":       {"type": "string"},
    },
}
_SEMANTIC_FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["category", "severity", "confidence", "referencedPO", "lineNumber", "explanation"],
    "properties": {
        "category":     {"type": "string", "enum": ["description_coherence", "date_plausibility", "packaging_plausibility", "pattern_detection"]},
        "severity":     {"type": "string", "enum": ["INFO", "WARNING", "CRITICAL"]},
        "confidence":   {"type": "number"},
        "referencedPO": {"type": ["string", "null"]},
        "lineNumber":   {"type": ["integer", "null"]},
        "explanation":  {"type": "string"},
    },
}
# Alignment sub-schemas. Strict mode requires every nested object to declare
# its full shape (additionalProperties: false, every property in `required`).
_TOLERANCES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["quantity_lower_pct", "quantity_upper_pct",
                 "price_lower_pct", "price_upper_pct",
                 "time_lower_days", "time_upper_days"],
    "properties": {
        "quantity_lower_pct": {"type": "number"},
        "quantity_upper_pct": {"type": "number"},
        "price_lower_pct":    {"type": "number"},
        "price_upper_pct":    {"type": "number"},
        "time_lower_days":    {"type": "integer"},
        "time_upper_days":    {"type": "integer"},
    },
}
_MATCHED_PO_SIDE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["quantity", "unitPrice", "currency", "unitOfMeasure",
                 "description", "requestedDeliveryDate", "tolerances"],
    "properties": {
        "quantity":              {"type": "number"},
        "unitPrice":             {"type": "number"},
        "currency":              {"type": "string"},
        "unitOfMeasure":         {"type": "string"},
        "description":           {"type": "string"},
        "requestedDeliveryDate": {"type": "string"},
        "tolerances":            _TOLERANCES_SCHEMA,
    },
}
_MATCHED_ASN_SIDE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["quantity", "unitPrice", "currency", "unitOfMeasure",
                 "description", "shipNoticeLineNumber"],
    "properties": {
        "quantity":              {"type": "number"},
        "unitPrice":             {"type": "number"},
        "currency":              {"type": "string"},
        "unitOfMeasure":         {"type": "string"},
        "description":           {"type": "string"},
        "shipNoticeLineNumber":  {"type": "integer"},
    },
}
_MATCHED_LINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineNumber", "buyerPartID", "po", "asn"],
    "properties": {
        "lineNumber":  {"type": "integer"},
        "buyerPartID": {"type": "string"},
        "po":          _MATCHED_PO_SIDE_SCHEMA,
        "asn":         _MATCHED_ASN_SIDE_SCHEMA,
    },
}
_UNMATCHED_LINE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineNumber", "buyerPartID", "reason"],
    "properties": {
        "lineNumber":  {"type": "integer"},
        "buyerPartID": {"type": "string"},
        "reason":      {"type": "string"},
    },
}
_VALIDATION_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["alignment", "ruleResults", "semanticFindings", "overallAssessment"],
    "properties": {
        "alignment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["portions"],
            "properties": {
                "portions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["referencedPO", "matchedLines", "unmatchedPO", "phantomASN"],
                        "properties": {
                            "referencedPO": {"type": "string"},
                            "matchedLines": {"type": "array", "items": _MATCHED_LINE_SCHEMA},
                            "unmatchedPO":  {"type": "array", "items": _UNMATCHED_LINE_SCHEMA},
                            "phantomASN":   {"type": "array", "items": _UNMATCHED_LINE_SCHEMA},
                        },
                    },
                },
            },
        },
        "ruleResults":      {"type": "array", "items": _RULE_RESULT_SCHEMA},
        "semanticFindings": {"type": "array", "items": _SEMANTIC_FINDING_SCHEMA},
        "overallAssessment": {"type": "string"},
    },
}


def _load_supplier_profile(asn_parsed: dict, po_baselines: dict) -> dict | None:
    """Lookup the matching supplier profile YAML.

    Resolves the vendor in this order (first one whose YAML exists wins):
      1. asn_parsed["supplierVendorID"]   — the explicit cXML VendorID credential
      2. asn_parsed["supplierNetworkID"]  — the cXML NetworkID credential (the
         anonymised corpus exposes this as the supplier identity, not VendorID)
      3. The first non-empty PO supplier networkID from po_baselines
      4. "UNKNOWN_SUPPLIER" sentinel (used for the missing-credential cohort)

    Returns the parsed profile dict (with `prompt_injection_line`) or None if
    no matching YAML exists. Profiles are generated by
    `Metodologia/scripts/build_supplier_profiles.py`.
    """
    if not USE_SUPPLIER_PROFILES:
        return None
    if not SUPPLIER_PROFILES_DIR.exists():
        return None

    candidates: list[str] = []
    v = (asn_parsed.get("supplierVendorID") or "").strip()
    if v:
        candidates.append(v)
    n = (asn_parsed.get("supplierNetworkID") or "").strip()
    if n:
        candidates.append(n)
    for b in (po_baselines or {}).values():
        cand = ((b.get("supplier") or {}).get("networkID") or "").strip()
        if cand and cand not in candidates:
            candidates.append(cand)
    candidates.append("UNKNOWN_SUPPLIER")

    yaml_path = None
    vendor = None
    for c in candidates:
        p = SUPPLIER_PROFILES_DIR / f"{c}.yaml"
        if p.exists():
            yaml_path = p
            vendor = c
            break
    if yaml_path is None:
        return None

    # Placebo arm: keep the matched supplier's framing (vendor_id label) but
    # source the history *content* from a different supplier, chosen
    # deterministically so a given real supplier always maps to the same donor.
    content_path = yaml_path
    placebo_source = None
    placebo_unavailable = False
    if RAG_PLACEBO:
        others = sorted(p for p in SUPPLIER_PROFILES_DIR.glob("*.yaml")
                        if p.stem != vendor)
        if others:
            idx = sum(ord(c) for c in (vendor or "")) % len(others)
            content_path = others[idx]
            placebo_source = content_path.stem
        else:
            placebo_unavailable = True  # only one profile exists; can't swap

    # Minimal YAML reader (profiles are flat, written by our own generator).
    profile: dict = {"_loaded_from": content_path.name, "vendor_id": vendor}
    if RAG_PLACEBO:
        profile["placebo"] = True
        profile["placebo_source"] = placebo_source
        profile["placebo_unavailable"] = placebo_unavailable
    for line in content_path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line or line.startswith("#") or line.startswith("  "):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        val = val.strip()
        if not val:
            continue
        try:
            profile[key.strip()] = json.loads(val)
        except json.JSONDecodeError:
            profile[key.strip()] = val
    return profile


def build_validation_prompt(po_baselines: dict, asn_parsed: dict, use_tools: bool = False) -> dict:
    rules_text = "\n".join(
        f"  - {r['id']} ({r['severity']}): {r['name']} -- {r['description']}"
        for r in RULE_DEFINITIONS
    )

    tool_instructions = ""
    if use_tools:
        tool_instructions = """

TOOL USE — MANDATORY FOR DETERMINISTIC RULES:
You have seven verification tools available. For the following rules you MUST
call the corresponding tool and write your verdict using the tool's `verdict`,
`severity`, and `explanation` fields, rather than computing the check inline:
  - R01 Shipped Quantity Match    -> verify_quantity_match(asn_qty, po_qty, lower_pct, upper_pct)
  - R02 Unit Price Match          -> verify_price_match(asn_price, po_price, lower_pct, upper_pct)
  - R03 Delivery Date Feasibility -> verify_date_arithmetic(asn_delivery_date, requested_delivery_date, lower_days, upper_days)
  - R06 Line Item Completeness    -> verify_line_completeness(po_line_numbers, asn_line_numbers)
  - R15 Total Value Reconciliation-> verify_line_total(unit_price, quantity, claimed_total)
  - R16 Partial Shipment Handling -> verify_partial_shipment(asn_qty, po_qty, lower_pct)
  - R19 ShipmentID Length         -> verify_shipment_id_length(shipment_id, max_length=35)

Pull the tolerance arguments from each PO line's `tolerances` block. Call one
tool per (rule, line) pair (header-level rules call once). Do not redo the
arithmetic inline; the tool's return value IS the deterministic answer.

R18 Duplicate ASN Detection is NOT exposed as a tool because it requires the
persistent set of previously-seen shipmentIDs. The pipeline will inject R18
post-hoc; do not attempt to evaluate it yourself."""

    system_prompt = f"""You are an expert procurement validation agent for Hilti.{tool_instructions}
Your task: given one or more Purchase Order (PO) baselines and a parsed Advanced Shipping Notice (ASN),
perform ALL of the following in a single pass:

1. **LINE ALIGNMENT**: For each ASN ShipNoticePortion, match ASN items to the corresponding PO's lines
   using (lineNumber, buyerPartID). Flag any PO lines missing from the ASN and any phantom ASN items.
   Note: One ASN can consolidate shipments for MULTIPLE POs -- each portion references one PO.

2. **RULE EVALUATION**: For each matched line pair, evaluate ALL 18 rules below.
   For header-level rules (R06, R07, R09, R10, R14, R17, R18), evaluate at document/portion level.

3. **SEMANTIC ANALYSIS**: Beyond the deterministic rules, check:
   - Description coherence: Does ASN item description semantically match PO item description?
   - Date plausibility: Are shipment/delivery dates realistic for the commodity, distances, quantities?
   - Packaging plausibility: Is the gross weight/volume reasonable for the items being shipped?
   - Pattern detection: Any cross-PO anomalies or suspicious patterns in the consolidated shipment?

RULES TO EVALUATE:
{rules_text}

DISPOSITION CALIBRATION (read carefully — affects how Stage 7 maps your findings to PASS/REVIEW/REJECT):
- A rule's `severity` field determines its disposition weight. CRITICAL fails -> REJECT, WARNING fails -> REVIEW, INFO fails -> PASS-with-note.
- Default to REVIEW unless you can identify at least one CRITICAL rule that fails. WARNING-only failures are NOT sufficient for REJECT — a single WARNING (or a cluster of WARNINGs without a CRITICAL anchor) should produce REVIEW so a buyer can decide.
- Do NOT over-classify findings as CRITICAL to "be safe". A spurious CRITICAL flips the disposition and wastes buyer time. Only mark CRITICAL when the registered severity in the rule list above is CRITICAL AND the violation is a hard, structural break (qty over band, missing PO, missing supplier, structurally invalid ID, line count divergent, etc.).
- For semantic findings (the `semanticFindings` array): use CRITICAL only if a buyer would block the goods receipt; WARNING for "this is unusual, ask before approving"; INFO for observations.
- R18 (Duplicate ASN) is a WARNING. A duplicate-ASN flag alone does NOT reject the document; the buyer needs to verify before blocking, because legitimate re-submissions exist.

IMPORTANT -- Arithmetic:
- When checking R01/R02/R15, show your calculations explicitly
- The Output Enforcer (Stage 6) will re-verify your arithmetic, so be precise.

Return ONLY valid JSON with this exact structure:
{{
  "alignment": {{
    "portions": [
      {{
        "referencedPO": "<orderID>",
        "matchedLines": [
          {{
            "lineNumber": <int>,
            "buyerPartID": "<string>",
            "po": {{ "quantity": <float>, "unitPrice": <float>, "currency": "<str>", "unitOfMeasure": "<str>", "description": "<str>", "requestedDeliveryDate": "<str>", "tolerances": {{...}} }},
            "asn": {{ "quantity": <float>, "unitPrice": <float>, "currency": "<str>", "unitOfMeasure": "<str>", "description": "<str>", "shipNoticeLineNumber": <int> }}
          }}
        ],
        "unmatchedPO": [ {{ "lineNumber": <int>, "buyerPartID": "<str>", "reason": "MISSING_FROM_ASN" }} ],
        "phantomASN": [ {{ "lineNumber": <int>, "buyerPartID": "<str>", "reason": "PHANTOM_IN_ASN" }} ]
      }}
    ]
  }},
  "ruleResults": [
    {{
      "rule_id": "<R01..R18>",
      "name": "<rule name>",
      "status": "<PASS|FAIL>",
      "severity": "<CRITICAL|WARNING|INFO>",
      "referencedPO": "<orderID or null for ASN-level>",
      "lineNumber": <int or null>,
      "detail": "<explanation with calculations>"
    }}
  ],
  "semanticFindings": [
    {{
      "category": "<description_coherence|date_plausibility|packaging_plausibility|pattern_detection>",
      "severity": "<INFO|WARNING|CRITICAL>",
      "confidence": <float 0.0-1.0>,
      "referencedPO": "<orderID or null>",
      "lineNumber": <int or null>,
      "explanation": "<detailed explanation>"
    }}
  ],
  "overallAssessment": "<brief summary>"
}}"""

    po_data = {}
    for oid, bl in po_baselines.items():
        po_data[oid] = {
            "orderID": bl["orderID"],
            "orderDate": bl.get("orderDate", ""),
            "orderVersion": bl.get("orderVersion", "1"),
            "currency": bl["currency"],
            "total": bl["total"],
            "incoterms": bl.get("incoterms", ""),
            "shipTo": bl.get("shipTo", {}),
            "supplier": bl.get("supplier", {}),
            "lineItems": bl["lineItems"],
        }

    payload_json = json.dumps({
        "po_baselines": po_data,
        "asn_parsed": asn_parsed,
    }, indent=2, ensure_ascii=False)
    if USE_STAGE5_FEW_SHOT:
        user_prompt = f"{VALIDATION_FEW_SHOT}\n\nCASE TO EVALUATE:\n{payload_json}"
    else:
        user_prompt = payload_json

    # Supplier-conditioned prompting (the "simple viable RAG"): if a profile
    # exists for this supplier, prepend its one-line historical-failure note to
    # the system prompt. The injection is gated on USE_SUPPLIER_PROFILES so it
    # can be ablated cleanly. See _load_supplier_profile() above and
    # Metodologia/scripts/build_supplier_profiles.py.
    supplier_profile = _load_supplier_profile(asn_parsed, po_baselines)
    if supplier_profile and supplier_profile.get("prompt_injection_line"):
        system_prompt = (
            f"SUPPLIER HISTORY ({supplier_profile['vendor_id']}): "
            f"{supplier_profile['prompt_injection_line']}\n\n"
            f"{system_prompt}"
        )

    context_metadata = {
        "shipmentID": asn_parsed.get("shipmentID", ""),
        "po_count": len(po_baselines),
        "po_orderIDs": list(po_baselines.keys()),
        "asn_portions": len(asn_parsed.get("portions", [])),
        "asn_total_items": sum(len(p.get("items", [])) for p in asn_parsed.get("portions", [])),
        "rules_injected": len(RULE_DEFINITIONS),
        "prompt_system_chars": len(system_prompt),
        "prompt_user_chars": len(user_prompt),
        "few_shot_enabled": USE_STAGE5_FEW_SHOT,
        "stage5_schema_enabled": USE_STAGE5_SCHEMA,
        "supplier_profile_used": supplier_profile["vendor_id"] if supplier_profile else None,
        "supplier_profile_n_prior": supplier_profile.get("n_prior_asns") if supplier_profile else None,
        "rag_placebo": bool(supplier_profile.get("placebo")) if supplier_profile else False,
        "rag_placebo_source": supplier_profile.get("placebo_source") if supplier_profile else None,
    }

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "context_metadata": context_metadata,
    }


# ══════════════════════════════════════════════════════════════
# Stage 5 — LLM Validation Agent
# ══════════════════════════════════════════════════════════════

def llm_validate(master_prompt: dict, use_tools: bool = False) -> dict:
    if use_tools:
        return _llm_validate_with_tools(master_prompt)
    schema = _VALIDATION_OUTPUT_SCHEMA if USE_STAGE5_SCHEMA else None
    raw_response = llm_call(
        system_prompt=master_prompt["system_prompt"],
        user_prompt=master_prompt["user_prompt"],
        temperature=0.0,
        schema=schema,
        schema_name="validation_output",
    )
    # Try parsing, with fallback for control characters the LLM may embed
    for attempt, text in enumerate([raw_response, raw_response.replace("\n", "\\n").replace("\r", "").replace("\t", "\\t")]):
        try:
            result = json.loads(text)
            if "error" in result and not result.get("ruleResults"):
                result["_llm_error"] = True
            return result
        except json.JSONDecodeError:
            if attempt == 0:
                # Second attempt: strip control chars inside string values
                import re
                cleaned = re.sub(r'[\x00-\x1f\x7f]', ' ', raw_response)
                try:
                    result = json.loads(cleaned)
                    print(f"  [WARNING] LLM JSON had control characters — cleaned successfully")
                    if "error" in result and not result.get("ruleResults"):
                        result["_llm_error"] = True
                    return result
                except json.JSONDecodeError:
                    continue
    # All attempts failed — return error structure instead of crashing
    print(f"  [ERROR] LLM returned unparseable JSON in Stage 5")
    print(f"  Raw response (first 500 chars): {raw_response[:500]}")
    return {
        "alignment": {"portions": []},
        "ruleResults": [],
        "semanticFindings": [],
        "overallAssessment": "LLM validation returned invalid JSON — manual review required",
        "_llm_error": True,
    }


def _llm_validate_with_tools(master_prompt: dict, max_iterations: int = 12) -> dict:
    """Stage 5 validation with in-generation function calling (tool_use mode).

    Passes the deterministic-verification tools (`_STAGE5_TOOLS`) to the model
    and runs the standard call/tool-result loop until the model returns a
    final non-tool response. The final response must be JSON and is parsed
    into the standard Stage 5 contract. Records every tool invocation in
    `tool_calls_log` so the smoke harness can verify the model actually used
    the tools."""
    messages = [
        {"role": "system", "content": master_prompt["system_prompt"]},
        {"role": "user",   "content": master_prompt["user_prompt"]},
    ]
    tool_calls_log = []
    final_text = ""
    try:
        for _iter in range(max_iterations):
            # Force a tool call on the first turn when enabled; "auto" afterwards
            # so the model can stop calling tools and emit the final verdict.
            _tc = "required" if (_TOOL_USE_FORCE_FIRST and _iter == 0) else "auto"
            _tk = dict(
                model=DEPLOYMENT,
                temperature=0.0,
                messages=messages,
                tools=_STAGE5_TOOLS,
                tool_choice=_tc,
            )
            if not _LLM_TOOLS_NO_RESPONSE_FORMAT:
                # Providers that reject tools+json_object (Mistral) skip this;
                # JSON is then obtained from the prompt instruction + parsing.
                _tk["response_format"] = {"type": "json_object"}
            if _LLM_SUPPORTS_SEED:
                _tk["seed"] = DET_SEED
            response = client.chat.completions.create(**_tk)
            msg = response.choices[0].message
            tcs = getattr(msg, "tool_calls", None) or []
            if not tcs:
                final_text = msg.content or ""
                break
            # Append assistant tool-call message before tool results.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    } for tc in tcs
                ],
            })
            for tc in tcs:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                impl = _STAGE5_TOOL_IMPLS.get(fn_name)
                if impl is None:
                    result = {"error": f"unknown tool: {fn_name}"}
                else:
                    try:
                        result = impl(**args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                tool_calls_log.append({
                    "stage": 5,
                    "rule_id": _TOOL_RULE_MAP.get(fn_name, "?"),
                    "name": fn_name,
                    "args": args,
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": json.dumps(result),
                })
        else:
            print(f"  [WARNING] tool_use loop exhausted after {max_iterations} iterations")
    except Exception as e:
        print(f"  [LLM ERROR tool_use] {type(e).__name__}: {e}")
        return {
            "alignment": {"portions": []},
            "ruleResults": [],
            "semanticFindings": [],
            "overallAssessment": f"LLM tool_use call failed: {type(e).__name__}: {e}",
            "tool_calls_log": tool_calls_log,
            "_llm_error": True,
        }

    _raw = final_text or ""
    # Robust JSON extraction: models without enforced json_object (e.g. Mistral in
    # tool_use) wrap the answer in markdown ```json ... ``` fences or prose. Try the
    # raw text, a control-char-stripped copy, a fence-stripped copy, and the first
    # balanced {...} substring. With json_object enforced (gpt) the first try wins.
    _fenced = re.sub(r'\s*```\s*$', '', re.sub(r'^\s*```(?:json)?\s*', '', _raw))
    _i, _j = _raw.find('{'), _raw.rfind('}')
    _brace = _raw[_i:_j + 1] if (_i != -1 and _j > _i) else ""
    for text in (_raw,
                 re.sub(r'[\x00-\x1f\x7f]', ' ', _raw),
                 _fenced,
                 _brace,
                 re.sub(r'[\x00-\x1f\x7f]', ' ', _brace)):
        if not (text or "").strip():
            continue
        try:
            result = json.loads(text)
            result["tool_calls_log"] = tool_calls_log
            return result
        except json.JSONDecodeError:
            continue
    print(f"  [ERROR] tool_use Stage 5 returned unparseable JSON")
    print(f"  Raw response (first 500 chars): {(final_text or '')[:500]}")
    return {
        "alignment": {"portions": []},
        "ruleResults": [],
        "semanticFindings": [],
        "overallAssessment": "LLM tool_use returned invalid JSON — manual review required",
        "tool_calls_log": tool_calls_log,
        "_llm_error": True,
    }


# ══════════════════════════════════════════════════════════════
# Stage 6 — Output Enforcer
# ══════════════════════════════════════════════════════════════

def _parse_date(s):
    if not s:
        return None
    s = re.sub(r"[+-]\d{2}:\d{2}$", "", s).replace("Z", "")
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


# ──────────────────────────────────────────────────────────────
# Per-rule deterministic classifiers (shared between enforce_output
# and compute_owned_rule_verdicts).  Pure functions, no I/O.
# ──────────────────────────────────────────────────────────────

def _classify_R01(po_qty: float, asn_qty: float,
                  lower_pct: float, upper_pct: float) -> dict:
    """R01 three-way: in_range PASS / under_ship PASS (queues R16) / over_ship FAIL."""
    lower_bound = po_qty * (1 - lower_pct / 100)
    upper_bound = po_qty * (1 + upper_pct / 100)
    if lower_bound <= asn_qty <= upper_bound:
        return {"status": "PASS", "ship_case": "in_range",
                "lower_bound": lower_bound, "upper_bound": upper_bound}
    if asn_qty < lower_bound:
        return {"status": "PASS", "ship_case": "under_ship",
                "lower_bound": lower_bound, "upper_bound": upper_bound}
    return {"status": "FAIL", "ship_case": "over_ship",
            "lower_bound": lower_bound, "upper_bound": upper_bound}


def _classify_R02(po_price: float, asn_price: float,
                  lower_pct: float, upper_pct: float) -> dict:
    """R02 asymmetric band on unit price."""
    lower_bound = po_price * (1 - lower_pct / 100)
    upper_bound = po_price * (1 + upper_pct / 100)
    status = "PASS" if lower_bound <= asn_price <= upper_bound else "FAIL"
    return {"status": status,
            "lower_bound": lower_bound, "upper_bound": upper_bound}


def _classify_R15(po_qty: float, po_price: float,
                  asn_qty: float, asn_price: float) -> dict:
    """R15 line-total reconciliation; tol = max(0.01, expected*0.0001)."""
    po_total = po_qty * po_price
    asn_total = asn_qty * asn_price
    tolerance = max(0.01, po_total * 0.0001)
    status = "PASS" if abs(po_total - asn_total) < tolerance else "FAIL"
    return {"status": status, "po_total": po_total,
            "asn_total": asn_total, "tolerance": tolerance}


def _classify_R03(requested_date_str: str, asn_delivery_str: str,
                  lower_days: int, upper_days: int) -> dict:
    """R03 window check: deliveryDate within
    [requestedDeliveryDate - lower_days, requestedDeliveryDate + upper_days]."""
    po_date = _parse_date(requested_date_str)
    asn_date = _parse_date(asn_delivery_str)
    if not po_date or not asn_date:
        return {"status": None, "window_start": None,
                "window_end": None, "asn_date": asn_date}
    window_start = po_date - timedelta(days=lower_days)
    window_end = po_date + timedelta(days=upper_days)
    status = "PASS" if window_start <= asn_date <= window_end else "FAIL"
    return {"status": status, "window_start": window_start,
            "window_end": window_end, "asn_date": asn_date}


def _classify_R19(shipment_id: str, max_length: int = 35) -> dict:
    """R19 ShipmentID Length."""
    length = len(shipment_id or "")
    status = "PASS" if length <= max_length else "FAIL"
    return {"status": status, "length": length, "max_length": max_length}


def _r18_shipment_id(asn_parsed: dict) -> str:
    """Deterministic shipmentID for the R18 duplicate key.

    R18 is a deterministic-owned rule, but asn_parsed["shipmentID"] is produced by
    the Stage-3 LLM parse, which can return '' or 'unknown'. If that value fed the
    duplicate key, the key would collapse to ('', PO-set) and any second ASN
    sharing the PO (even a genuinely new shipment) would be flagged as a duplicate.
    We therefore read the shipmentID straight from the raw cXML attribute, which is
    always present in asn_parsed["raw_xml"]."""
    sid = (asn_parsed.get("shipmentID") or "").strip()
    if sid and sid.lower() != "unknown":
        return sid
    raw = asn_parsed.get("raw_xml") or ""
    m = re.search(r'shipmentID\s*=\s*"([^"]+)"', raw)
    return m.group(1) if m else sid


def _classify_R18(shipment_id: str, ref_pos: tuple,
                  seen_asns: set, mutate: bool) -> dict:
    """R18 Duplicate ASN. If mutate=True and not seen, adds to set + persists."""
    asn_key = (shipment_id, ref_pos)
    if DISABLE_R18:
        return {"status": "PASS", "is_duplicate": False, "asn_key": asn_key}
    is_duplicate = asn_key in seen_asns
    if not is_duplicate and mutate:
        seen_asns.add(asn_key)
        _save_seen_asns(seen_asns)
    status = "FAIL" if is_duplicate else "PASS"
    return {"status": status, "is_duplicate": is_duplicate, "asn_key": asn_key}


def _classify_R09(asn_parsed: dict, po_baselines: dict) -> dict:
    """R09 Supplier ID Validation (deterministic).

    Compares same-credential-type credentials only. Cross-credential
    comparison (ASN VendorID vs PO NetworkID) is a category error: VendorID
    is the buyer's internal ANID and is not published on POs, so the two IDs
    will never be equal even when they refer to the same supplier.

    Logic:
      Case 1  ASN has neither VendorID nor NetworkID
              -> FAIL (missing credential, supplier unidentifiable)
      Case 2  ASN has NetworkID and PO has NetworkID
              -> compare them. PASS if equal; FAIL if mismatch.
      Case 3  ASN has only VendorID (no NetworkID), PO has NetworkID only
              -> PASS (no arithmetic check possible; trust upstream PO
              selection vetted the supplier).

    The LLM owns R09 as a primary check; this deterministic version exists
    to enforce the same-type-only invariant that the LLM occasionally
    violates by surface-comparing the two ID strings.
    """
    asn_vendor = (asn_parsed.get("supplierVendorID") or "").strip()
    asn_network = (asn_parsed.get("supplierNetworkID") or "").strip()
    po_network_ids = {
        (b.get("supplier") or {}).get("networkID", "").strip()
        for b in (po_baselines or {}).values()
    }
    po_network_ids.discard("")

    # Case 1: nothing to compare on either credential type
    if not asn_vendor and not asn_network:
        return {"status": "FAIL",
                "asn_vendor": "", "asn_network": "",
                "po_network_ids": sorted(po_network_ids),
                "reason": "missing_credential",
                "detail": "ASN has no VendorID nor NetworkID credential "
                          "— supplier cannot be identified."}

    # Case 2: ASN NetworkID is the canonical signal — compare against PO NetworkID
    if asn_network:
        if not po_network_ids:
            return {"status": "PASS",
                    "asn_vendor": asn_vendor, "asn_network": asn_network,
                    "po_network_ids": [],
                    "reason": "po_has_no_network_id",
                    "detail": f"ASN NetworkID {asn_network!r} present but PO has no "
                              "NetworkID to compare against; treat as PASS "
                              "(upstream PO selection vetted the supplier)."}
        if asn_network in po_network_ids:
            return {"status": "PASS",
                    "asn_vendor": asn_vendor, "asn_network": asn_network,
                    "po_network_ids": sorted(po_network_ids),
                    "reason": "network_match",
                    "detail": f"ASN NetworkID {asn_network!r} matches PO supplier NetworkID."}
        return {"status": "FAIL",
                "asn_vendor": asn_vendor, "asn_network": asn_network,
                "po_network_ids": sorted(po_network_ids),
                "reason": "network_mismatch",
                "detail": f"ASN NetworkID {asn_network!r} does not match any PO "
                          f"supplier NetworkID in {sorted(po_network_ids)}."}

    # Case 3: ASN has only VendorID (buyer-internal ANID). PO does not publish
    # VendorID, so no cross-comparison is meaningful. Trust upstream PO
    # selection as the identity anchor.
    return {"status": "PASS",
            "asn_vendor": asn_vendor, "asn_network": "",
            "po_network_ids": sorted(po_network_ids),
            "reason": "vendor_only_no_arithmetic_check",
            "detail": f"ASN carries only VendorID {asn_vendor!r} (buyer-internal "
                      "ANID); PO does not publish VendorID, so no arithmetic "
                      "supplier-identity check is possible — upstream PO selection "
                      "is the trust anchor."}


# UoM whitelists used by R20 + R21 deterministic checks. Sources:
#   - UN/CEFACT Recommendation 20 (subset Hilti uses in cXML)
#   - Buyer Guide §7.1: "KG used for VOLUME" → mismatch
_MASS_UOMS = {"KGM", "GRM", "LBR", "ONZ", "TNE", "MGM", "STN", "LTN"}
_VOLUME_UOMS = {"MTQ", "LTR", "MLT", "INQ", "FTQ", "GLI", "GLL", "DMQ", "CMQ", "M3"}


def _classify_R20(asn_parsed: dict) -> dict:
    """R20 Weight Plausibility (deterministic).

    The Buyer Guide §4.2/§4.5/§4.6 enumerates two failure modes:
      (a) gross <= net      → data-entry inversion
      (b) gross > net * 10  → data-entry magnitude error

    R20 fires if EITHER (a) or (b) is true.  When both weights are absent,
    we return PASS — the absence is R21's territory, not R20's.

    Returns a dict with status / reason / values for the enforcer log.
    """
    pkg = asn_parsed.get("packaging") or {}
    gross = (pkg.get("grossWeight") or {}).get("quantity")
    net = (pkg.get("netWeight") or {}).get("quantity")
    # Fallback: if Stage 3 dropped a weight (e.g. schema/parse drift), recover
    # it straight from the raw cXML so R20 still has its inputs. R21 uses the
    # same raw_xml escape hatch for the packaging-type marker.
    if not gross or not net:
        raw = asn_parsed.get("raw_xml") or ""
        if raw:
            def _dim_qty(dim_type: str):
                m = re.search(
                    rf'<Dimension[^>]*\btype="{re.escape(dim_type)}"[^>]*\bquantity="([^"]+)"',
                    raw)
                return _parse_money(m.group(1)) if m else None
            if gross in (None, 0):
                gross = _dim_qty("grossWeight")
            if net in (None, 0):
                net = _dim_qty("netWeight")
    try:
        gross_f = float(gross) if gross is not None else None
        net_f = float(net) if net is not None else None
    except (TypeError, ValueError):
        return {"status": "PASS", "reason": "non_numeric_weights",
                "detail": f"Weights present but non-numeric (gross={gross!r}, net={net!r})."}
    # quantity 0 is the parser's absent-sentinel (see schema note), not a real
    # weight — treat it as missing so R20 doesn't false-fire on the common
    # no-netWeight ASN. Presence/UoM completeness is R21's job.
    if not gross_f or not net_f:
        return {"status": "PASS", "reason": "weights_absent_R21_territory",
                "detail": "grossWeight or netWeight missing/zero — handled by R21."}
    if gross_f <= net_f:
        return {"status": "FAIL", "reason": "gross_not_greater_than_net",
                "gross": gross_f, "net": net_f,
                "detail": f"grossWeight {gross_f} not strictly greater than netWeight {net_f}; "
                          "data-entry inversion (Buyer Guide §4.5/§4.6)."}
    if gross_f > net_f * 10:
        return {"status": "FAIL", "reason": "gross_more_than_10x_net",
                "gross": gross_f, "net": net_f, "ratio": round(gross_f / net_f, 2),
                "detail": f"grossWeight {gross_f} > netWeight {net_f} * 10 "
                          f"(ratio {gross_f/net_f:.1f}); data-entry magnitude error "
                          "(Buyer Guide §4.2)."}
    return {"status": "PASS", "reason": "ratio_within_band",
            "gross": gross_f, "net": net_f,
            "detail": f"grossWeight {gross_f} > netWeight {net_f} and within 10x band."}


def _classify_R21(asn_parsed: dict) -> dict:
    """R21 Mandatory Packaging Fields (deterministic).

    Buyer Guide §4.7, §4.8, §5.2, §7.1/§7.2 require:
      (a) netWeight present with a valid mass UoM
      (b) grossWeight present with a valid mass UoM
      (c) at least one packaging type (Extrinsic 'nccExtrinsic*' or similar)
      (d) when grossVolume is present, its UoM must be a volume unit
          (KG for VOLUME = guide §7.1 mismatch)

    Returns a single rolled-up dict listing every failing sub-check so the
    buyer (and the enforcer log) sees the full picture in one place.
    """
    pkg = asn_parsed.get("packaging") or {}
    issues: list[str] = []
    notes: list[str] = []

    # Net weight is sometimes provided at the line level (ShipNoticeItemDetail/
    # Dimension type="unitNetWeight") rather than in Packaging. If so, the net
    # weight IS present — only its placement is non-canonical. Treat that as a
    # placement note (not a completeness failure) so we don't over-review a
    # shipment whose net weight is actually declared. Genuinely-absent net
    # weight (no Packaging.netWeight AND no line unitNetWeight) still fails.
    line_net_present = any(
        float(it.get("unitNetWeight") or 0) > 0
        for p in (asn_parsed.get("portions") or [])
        for it in (p.get("items") or [])
    )
    if not line_net_present:
        # Fallback: scan raw cXML for a line-level unitNetWeight Dimension with
        # a positive quantity (covers the case where Stage 3 didn't surface it).
        raw_xml = asn_parsed.get("raw_xml") or ""
        line_net_present = any(
            _parse_money(q) > 0
            for q in re.findall(
                r'<Dimension[^>]*\btype="unitNetWeight"[^>]*\bquantity="([^"]+)"',
                raw_xml)
        )

    def _check_dim(name: str, expected_set: set, label: str,
                   line_fallback: bool = False):
        d = pkg.get(name) or {}
        qty = d.get("quantity")
        uom = (d.get("uom") or "").upper().strip()
        missing = qty in (None, "", 0, "0")
        if missing and line_fallback and line_net_present:
            notes.append(f"{name} provided at line level (unitNetWeight), "
                         "not in Packaging (R11 placement note)")
            return
        if missing:
            issues.append(f"{name} value missing")
        if not uom:
            issues.append(f"{name} unit-of-measure missing")
        elif uom not in expected_set:
            issues.append(f"{name} uses non-{label} UoM '{uom}' (expected one of {sorted(expected_set)[:5]}...)")

    _check_dim("netWeight",   _MASS_UOMS,   "mass", line_fallback=True)
    _check_dim("grossWeight", _MASS_UOMS,   "mass")

    # grossVolume is optional; only check UoM if present
    if "grossVolume" in pkg:
        gv = pkg["grossVolume"] or {}
        uom = (gv.get("uom") or "").upper().strip()
        if uom and uom not in _VOLUME_UOMS:
            issues.append(f"grossVolume uses non-volume UoM '{uom}' (expected MTQ/LTR/M3/...)")

    # Packaging type ("CARTON", "EUPAL", ...) lives in the raw cXML under
    # <Extrinsic name="nccExtrinsic*">. Stage 3 (LLM normalisation) doesn't
    # expose this consistently, so we scan the raw_xml when present. If the
    # raw_xml is unavailable we DON'T fail — the LLM still owns this leg.
    raw = asn_parsed.get("raw_xml") or ""
    if raw:
        if "nccExtrinsic" not in raw and "<Packaging" in raw:
            # Packaging block present but no Extrinsic packaging-type marker
            issues.append("no packaging type (Extrinsic nccExtrinsic*) present")

    if issues:
        return {"status": "FAIL", "reason": "packaging_incomplete",
                "issues": issues,
                "detail": "; ".join(issues + notes)}
    return {"status": "PASS", "reason": "packaging_complete",
            "notes": notes,
            "detail": "netWeight + grossWeight with valid mass UoMs"
                      + (" (netWeight at line level)" if notes else "")
                      + (" and packaging type marker present" if "nccExtrinsic" in (asn_parsed.get("raw_xml") or "") else "")
                      + "."}


def _portion_coverage(po_baselines: dict, asn_parsed: dict) -> dict:
    """Per-PO line-coverage summary used by R06 logic."""
    coverage = {}
    for portion in asn_parsed.get("portions", []):
        pid = portion.get("referencedPO", "")
        if not pid:
            continue
        asn_lines = {it["lineNumber"] for it in portion.get("items", [])}
        po_lines = {li["lineNumber"]
                    for li in po_baselines.get(pid, {}).get("lineItems", [])}
        coverage[pid] = {"asn": asn_lines, "po": po_lines}
    return coverage


def _classify_R06(po_baselines: dict, asn_parsed: dict) -> dict:
    """R06 Line Item Completeness (deterministic).

    Compares PO line numbers vs ASN line numbers for every referenced PO via
    `_portion_coverage`. Three outcomes:
      - FULL_MISS  PO has lines but ASN portion has zero coverage
                   -> CRITICAL  (true completeness violation)
      - PARTIAL    PO has lines, ASN covers some, others missing
                   -> WARNING  (legitimate partial delivery)
      - COMPLETE   ASN covers every PO line referenced
                   -> PASS

    Returns the same shape as the partition-mode R06 verdict so callers can
    inject/override the result symmetrically across enforcer modes.
    """
    coverage = _portion_coverage(po_baselines, asn_parsed)
    full_misses, partials = [], []
    for pid, cov in coverage.items():
        if not cov["po"]:
            continue
        missing = sorted(cov["po"] - cov["asn"])
        if not missing:
            continue
        if cov["asn"]:
            partials.append((pid, missing, sorted(cov["asn"])))
        else:
            full_misses.append((pid, missing))

    if full_misses:
        pid, missing = full_misses[0]
        return {
            "status": "FAIL", "severity": "CRITICAL",
            "referencedPO": pid, "lineNumber": None,
            "reason": "full_miss",
            "detail": (f"R06: PO {pid} has no ASN coverage; "
                       f"missing lines {missing}"),
        }
    if partials:
        pid, missing, present = partials[0]
        return {
            "status": "FAIL", "severity": "WARNING",
            "referencedPO": pid, "lineNumber": None,
            "reason": "partial_delivery",
            "detail": (f"R06: partial delivery on {len(partials)} PO(s); "
                       f"first: PO {pid} covers {present} missing {missing} "
                       f"[PARTIAL DELIVERY]"),
        }
    return {
        "status": "PASS", "severity": _RULE_SEVERITY.get("R06", "CRITICAL"),
        "referencedPO": None, "lineNumber": None,
        "reason": "complete",
        "detail": "R06: all referenced PO lines covered",
    }


def _classify_R17(po_baselines: dict, asn_parsed: dict) -> dict:
    """R17 Multi-PO Consolidation / cross-PO item contamination (deterministic).

    Each ShipNoticePortion references exactly one PO, and an item's only PO
    association *is* the portion it sits in — so a moved item heals itself
    structurally and can only be detected against an external reference: the
    PO baselines the pipeline already loads.

    For every portion we ask, per item: does this item belong to its OWN
    portion's PO?  An item "belongs to" a PO when its buyerPartID matches a
    line on that PO (the Stage 5 alignment key is (lineNumber, buyerPartID)),
    falling back to lineNumber when buyerPartID is absent.  If an item does
    not belong to its own PO but DOES belong to a *different* referenced PO,
    that is cross-PO contamination -> FAIL.  Items matching neither are
    phantoms (R06/alignment territory, not R17) and are ignored here.

    Single-PO ASNs cannot contaminate (PASS).  POs absent from the baseline
    set are skipped — there is nothing to check them against.
    """
    portions = asn_parsed.get("portions") or []
    referenced = [p.get("referencedPO", "") for p in portions
                  if p.get("referencedPO")]
    # Need at least two distinct referenced POs that we hold baselines for.
    known = [pid for pid in referenced if pid in po_baselines]
    if len(set(known)) < 2:
        return {"status": "PASS", "reason": "single_po_or_no_baseline",
                "detail": "R17: fewer than two referenced POs with baselines; "
                          "cross-PO contamination not applicable."}

    def _belongs(pid: str, item: dict) -> bool:
        lines = po_baselines.get(pid, {}).get("lineItems", [])
        bpid = (item.get("buyerPartID") or "").strip()
        if bpid:
            return any((li.get("buyerPartID") or "").strip() == bpid for li in lines)
        ln = item.get("lineNumber")
        return any(li.get("lineNumber") == ln for li in lines)

    for portion in portions:
        own = portion.get("referencedPO", "")
        if own not in po_baselines:
            continue
        for item in portion.get("items", []):
            if _belongs(own, item):
                continue
            # Not on its own PO — is it on another referenced PO?
            for other in known:
                if other == own:
                    continue
                if _belongs(other, item):
                    ident = (item.get("buyerPartID")
                             or f"line {item.get('lineNumber')}")
                    return {
                        "status": "FAIL", "reason": "cross_po_item",
                        "referencedPO": own,
                        "lineNumber": item.get("lineNumber"),
                        "detail": (f"R17: item {ident} is filed under PO {own} but "
                                   f"matches PO {other} instead — cross-PO "
                                   f"contamination (item placed in the wrong "
                                   f"ShipNoticePortion)."),
                    }
    return {"status": "PASS", "reason": "no_cross_po_contamination",
            "detail": "R17: every item matches its own portion's PO."}


def enforce_output(master_report: dict, po_baselines: dict, asn_parsed: dict) -> dict:
    enforcer_log = []

    # 1. Schema Validation
    required_keys = ["alignment", "ruleResults", "semanticFindings", "overallAssessment"]
    for key in required_keys:
        if key not in master_report:
            enforcer_log.append({"type": "SCHEMA_ERROR", "detail": f"Missing required key: {key}"})
            if key == "alignment":
                master_report[key] = {"portions": []}
            elif key in ("ruleResults", "semanticFindings"):
                master_report[key] = []
            else:
                master_report[key] = "N/A"

    # 2. Severity capping
    # Each rule has a single registered severity (RULE_DEFINITIONS / Table A.1).
    # The LLM is free to invent its own severity per emitted ruleResult; we
    # clamp it back to the registered value here so downstream consumers
    # (Decision Router, audit log, evaluation) see a consistent severity model
    # regardless of LLM drift.
    _cap_rule_severities(master_report.get("ruleResults", []), enforcer_log)

    # 3. Build lookups
    po_line_lookup = {}
    for oid, bl in po_baselines.items():
        for li in bl["lineItems"]:
            po_line_lookup[(oid, li["lineNumber"])] = li

    asn_line_lookup = {}
    for portion in asn_parsed.get("portions", []):
        po_id = portion.get("referencedPO", "")
        for item in portion.get("items", []):
            asn_line_lookup[(po_id, item["lineNumber"])] = item

    # 4. Arithmetic Re-verification
    under_shipped_lines = []  # collected for R16 escalation after the loop
    for rule_result in master_report.get("ruleResults", []):
        rule_id = rule_result.get("rule_id", "")
        line_num = rule_result.get("lineNumber")
        ref_po = rule_result.get("referencedPO", "")

        if rule_id not in ("R01", "R02", "R15", "R03") or line_num is None or not ref_po:
            continue

        po_line = po_line_lookup.get((ref_po, line_num))
        asn_line = asn_line_lookup.get((ref_po, line_num))
        if not po_line or not asn_line:
            continue

        tol = po_line.get("tolerances", {})

        if rule_id == "R01":
            po_qty = po_line["quantity"]
            asn_qty = asn_line["quantity"]
            lower_pct = tol.get("quantity_lower_pct", 0)
            upper_pct = tol.get("quantity_upper_pct", 0)
            r01 = _classify_R01(po_qty, asn_qty, lower_pct, upper_pct)
            python_status = r01["status"]
            ship_case = r01["ship_case"]
            lower_bound = r01["lower_bound"]
            upper_bound = r01["upper_bound"]
            if ship_case == "under_ship":
                under_shipped_lines.append({
                    "referencedPO": ref_po,
                    "lineNumber": line_num,
                    "po_qty": po_qty,
                    "asn_qty": asn_qty,
                    "remaining": po_qty - asn_qty,
                })

            if python_status != rule_result["status"]:
                enforcer_log.append({
                    "type": "ARITHMETIC_OVERRIDE", "rule_id": "R01",
                    "referencedPO": ref_po, "lineNumber": line_num,
                    "llm_said": rule_result["status"], "python_says": python_status,
                    "ship_case": ship_case,
                    "detail": (f"qty {asn_qty} vs range [{lower_bound}, {upper_bound}] "
                               f"({ship_case})"),
                })
                rule_result["status"] = python_status
                rule_result["detail"] += (
                    f" [ENFORCER OVERRIDE -> {python_status} ({ship_case})]"
                )
            if ship_case == "under_ship":
                enforcer_log.append({
                    "type": "UNDER_SHIP_ROUTED_TO_R16",
                    "rule_id": "R01",
                    "referencedPO": ref_po, "lineNumber": line_num,
                    "detail": (f"Under-shipment qty {asn_qty} of PO qty {po_qty} "
                               f"(remaining {po_qty - asn_qty}) -> R16 WARNING"),
                })

        elif rule_id == "R02":
            po_price = po_line["unitPrice"]
            asn_price = asn_line["unitPrice"]
            lower_pct = tol.get("price_lower_pct", 0)
            upper_pct = tol.get("price_upper_pct", 0)
            r02 = _classify_R02(po_price, asn_price, lower_pct, upper_pct)
            python_status = r02["status"]
            lower_bound = r02["lower_bound"]
            upper_bound = r02["upper_bound"]
            if python_status != rule_result["status"]:
                enforcer_log.append({
                    "type": "ARITHMETIC_OVERRIDE", "rule_id": "R02",
                    "referencedPO": ref_po, "lineNumber": line_num,
                    "llm_said": rule_result["status"], "python_says": python_status,
                    "detail": f"price {asn_price} vs range [{lower_bound}, {upper_bound}]",
                })
                rule_result["status"] = python_status
                rule_result["detail"] += f" [ENFORCER OVERRIDE -> {python_status}]"

        elif rule_id == "R15":
            r15 = _classify_R15(po_line["quantity"], po_line["unitPrice"],
                                 asn_line["quantity"], asn_line["unitPrice"])
            python_status = r15["status"]
            po_total = r15["po_total"]
            asn_total = r15["asn_total"]
            if python_status != rule_result["status"]:
                enforcer_log.append({
                    "type": "ARITHMETIC_OVERRIDE", "rule_id": "R15",
                    "referencedPO": ref_po, "lineNumber": line_num,
                    "llm_said": rule_result["status"], "python_says": python_status,
                    "detail": f"PO total={po_total:.2f}, ASN total={asn_total:.2f}",
                })
                rule_result["status"] = python_status
                rule_result["detail"] += f" [ENFORCER OVERRIDE -> {python_status}]"

        elif rule_id == "R03":
            requested_date = po_line.get("requestedDeliveryDate", "")
            asn_delivery = asn_parsed.get("deliveryDate", "")
            tol = po_line.get("tolerances", {})
            lower_days = tol.get("time_lower_days", 0)
            upper_days = tol.get("time_upper_days", 0)
            r03 = _classify_R03(requested_date, asn_delivery, lower_days, upper_days)
            if r03["status"] is not None:
                python_status = r03["status"]
                if python_status != rule_result["status"]:
                    enforcer_log.append({
                        "type": "DATE_OVERRIDE", "rule_id": "R03",
                        "referencedPO": ref_po, "lineNumber": line_num,
                        "llm_said": rule_result["status"], "python_says": python_status,
                        "detail": (f"deliveryDate {r03['asn_date']} vs window "
                                   f"[{r03['window_start']}, {r03['window_end']}]"),
                    })
                    rule_result["status"] = python_status
                    rule_result["detail"] += f" [ENFORCER OVERRIDE -> {python_status}]"

    # 3b. R16 Partial Shipment Escalation — distinct REVIEW path vs over-ship REJECT.
    # Under-shipment is routed here (R01 stays PASS). R16 is INFO by default; we
    # find-or-inject a FAIL at WARNING severity so Stage 7 computes REVIEW.
    rule_results = master_report.get("ruleResults", [])
    for us in under_shipped_lines:
        ref_po = us["referencedPO"]
        line_num = us["lineNumber"]
        detail_msg = (f"Partial shipment: ASN qty {us['asn_qty']} of PO qty "
                      f"{us['po_qty']} (remaining {us['remaining']})")
        match = next(
            (r for r in rule_results
             if r.get("rule_id") == "R16"
             and r.get("referencedPO") == ref_po
             and r.get("lineNumber") == line_num),
            None,
        )
        if match is None:
            rule_results.append({
                "rule_id": "R16", "name": "Partial Shipment Handling",
                "status": "FAIL", "severity": "WARNING",
                "referencedPO": ref_po, "lineNumber": line_num,
                "detail": f"{detail_msg} [ENFORCER INJECTED -> WARNING]",
            })
            enforcer_log.append({
                "type": "R16_PARTIAL_SHIPMENT_INJECTED",
                "rule_id": "R16",
                "referencedPO": ref_po, "lineNumber": line_num,
                "detail": detail_msg,
            })
        else:
            prev_status = match.get("status")
            prev_severity = match.get("severity", "INFO")
            match["status"] = "FAIL"
            match["severity"] = "WARNING"
            match["detail"] = (f"{detail_msg} "
                               f"[ENFORCER: escalated {prev_severity}/{prev_status} "
                               f"-> WARNING/FAIL]")
            enforcer_log.append({
                "type": "R16_PARTIAL_SHIPMENT_ESCALATED",
                "rule_id": "R16",
                "referencedPO": ref_po, "lineNumber": line_num,
                "llm_said": f"{prev_severity}/{prev_status}",
                "python_says": "WARNING/FAIL",
                "detail": detail_msg,
            })

    # 3c. R06 Line Item Completeness — deterministic verdict via _classify_R06.
    # Three regimes are handled symmetrically with the LLM's emission:
    #   (a) LLM did NOT emit R06 -> INJECT the deterministic verdict.
    #   (b) LLM emitted R06=PASS but deterministic says FAIL -> OVERRIDE.
    #   (c) LLM emitted R06=FAIL with wrong severity -> CORRECT to deterministic
    #       severity (CRITICAL for full miss, WARNING for partial delivery).
    # Replaces the previous downgrade-only logic. The buyer expected R06 to
    # fire on most AT-cluster cases (P-38..P-43); the LLM was missing it
    # because in deterministic-mode it has no way to know R06 is owned.
    r06 = _classify_R06(po_baselines, asn_parsed)
    existing_r06_fails = [r for r in rule_results
                          if r.get("rule_id") == "R06" and r.get("status") == "FAIL"]

    if r06["status"] == "FAIL":
        if not existing_r06_fails:
            # (a) Inject
            rule_results.append({
                "rule_id": "R06", "name": "Line Item Completeness",
                "status": "FAIL", "severity": r06["severity"],
                "referencedPO": r06["referencedPO"],
                "lineNumber": r06["lineNumber"],
                "detail": r06["detail"] + " [ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R06",
                "reason": r06["reason"], "detail": r06["detail"],
            })
        else:
            # (c) Correct severity on existing LLM emission(s) — apply the
            # deterministic severity to ALL R06 FAILs the LLM emitted (which
            # may be per-line or aggregated). Same severity for all.
            for r in existing_r06_fails:
                if r.get("severity") != r06["severity"]:
                    prev = r.get("severity", "?")
                    r["severity"] = r06["severity"]
                    r["detail"] = (
                        (r.get("detail") or "")
                        + f" [ENFORCER: {prev} -> {r06['severity']} "
                          f"({r06['reason']})]"
                    )
                    enforcer_log.append({
                        "type": "R06_SEVERITY_CORRECTED",
                        "rule_id": "R06",
                        "referencedPO": r.get("referencedPO"),
                        "lineNumber": r.get("lineNumber"),
                        "llm_said": prev,
                        "python_says": r06["severity"],
                        "reason": r06["reason"],
                    })
    else:
        # Deterministic R06 PASS: any LLM-emitted R06 FAIL is wrong; override.
        for r in existing_r06_fails:
            llm_status = r.get("status")
            r["status"] = "PASS"
            r["severity"] = r06["severity"]
            r["detail"] = (r06["detail"]
                           + f" [ENFORCER OVERRIDE -> PASS, was {llm_status}]")
            enforcer_log.append({
                "type": "ARITHMETIC_OVERRIDE",
                "rule_id": "R06",
                "llm_said": llm_status,
                "python_says": "PASS",
                "reason": r06["reason"],
                "detail": r06["detail"],
            })

    # 5. Duplicate ASN Detection (R18)
    shipment_id = _r18_shipment_id(asn_parsed)
    ref_pos = tuple(sorted(
        p.get("referencedPO", "") for p in asn_parsed.get("portions", [])
    ))
    r18 = _classify_R18(shipment_id, ref_pos, _seen_asns, mutate=True)
    if r18["is_duplicate"]:
        r18_found = any(r["rule_id"] == "R18" and r["status"] == "FAIL"
                        for r in master_report.get("ruleResults", []))
        if not r18_found:
            master_report["ruleResults"].append({
                "rule_id": "R18", "name": "Duplicate ASN Detection",
                "status": "FAIL", "severity": _RULE_SEVERITY["R18"],
                "referencedPO": None, "lineNumber": None,
                "detail": f"Duplicate ASN: shipmentID={shipment_id} [ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R18",
                "detail": f"LLM missed duplicate ASN {shipment_id}",
            })

    # 4b. Cap semantic-finding severity by analogous rule severity.
    # The LLM sometimes emits CRITICAL semantic findings for categories that
    # duplicate a WARNING rule (e.g. date_plausibility duplicates R03 WARNING).
    # Each such finding inflates disposition counts in Stage 7 and can flip
    # REVIEW→REJECT. Clamp the severity down to match the owning rule.
    _SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
    _SEM_CATEGORY_RULE_SEVERITY = {
        "date_plausibility": "WARNING",       # R03
        "description_coherence": "WARNING",   # R12
    }
    for finding in master_report.get("semanticFindings", []):
        category = finding.get("category", "")
        cap = _SEM_CATEGORY_RULE_SEVERITY.get(category)
        if not cap:
            continue
        current = finding.get("severity", "INFO")
        if _SEVERITY_RANK.get(current, 0) > _SEVERITY_RANK[cap]:
            enforcer_log.append({
                "type": "SEMANTIC_SEVERITY_CAPPED",
                "category": category,
                "llm_said": current, "capped_to": cap,
                "detail": f"Semantic category '{category}' capped to match analogous rule severity",
            })
            finding["severity"] = cap

    # 5b. Supplier ID Validation (R09) — fires when ASN has no VendorID
    # credential at all, or when its VendorID doesn't match any PO supplier.
    # The LLM has no signal for the missing-credential case (nothing to read),
    # so the deterministic enforcer injects R09=FAIL the same way R18/R19 are
    # injected. If the LLM already produced an R09=FAIL, the enforcer-injected
    # entry is suppressed.
    r09 = _classify_R09(asn_parsed, po_baselines)
    if r09["status"] == "FAIL":
        r09_found = any(r.get("rule_id") == "R09" and r.get("status") == "FAIL"
                        for r in master_report.get("ruleResults", []))
        if not r09_found:
            master_report["ruleResults"].append({
                "rule_id": "R09", "name": "Supplier ID Validation",
                "status": "FAIL", "severity": _RULE_SEVERITY["R09"],
                "referencedPO": None, "lineNumber": None,
                "detail": r09["detail"] + " [ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R09",
                "reason": r09["reason"],
                "detail": r09["detail"],
            })

    # 5c. Weight Plausibility (R20) — strict gross > net AND gross <= net*10.
    # The Stage 5 LLM checks R20 too; the deterministic verdict overrides
    # only when they disagree, mirroring the R01/R02 arithmetic-override
    # pattern. When the LLM didn't emit R20 at all, we inject FAIL only.
    r20 = _classify_R20(asn_parsed)
    existing_r20 = next((r for r in master_report.get("ruleResults", [])
                        if r.get("rule_id") == "R20"), None)
    if r20["status"] == "FAIL":
        if existing_r20 is None:
            master_report["ruleResults"].append({
                "rule_id": "R20", "name": "Weight Plausibility",
                "status": "FAIL", "severity": _RULE_SEVERITY["R20"],
                "referencedPO": None, "lineNumber": None,
                "detail": r20["detail"] + " [ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R20",
                "reason": r20["reason"], "detail": r20["detail"],
            })
        elif existing_r20.get("status") != "FAIL":
            llm_said = existing_r20.get("status")
            existing_r20["status"] = "FAIL"
            existing_r20["detail"] = r20["detail"] + " [ENFORCER OVERRIDE]"
            enforcer_log.append({
                "type": "ARITHMETIC_OVERRIDE", "rule_id": "R20",
                "llm_said": llm_said, "python_says": "FAIL",
                "reason": r20["reason"], "detail": r20["detail"],
            })

    # 5d. Mandatory Packaging Fields (R21) — weights with valid mass UoM,
    # grossVolume (if present) with a volume UoM, packaging type marker.
    # Same injection-or-override pattern as R20.
    r21 = _classify_R21(asn_parsed)
    existing_r21 = next((r for r in master_report.get("ruleResults", [])
                        if r.get("rule_id") == "R21"), None)
    if r21["status"] == "FAIL":
        if existing_r21 is None:
            master_report["ruleResults"].append({
                "rule_id": "R21", "name": "Mandatory Packaging Fields",
                "status": "FAIL", "severity": _RULE_SEVERITY["R21"],
                "referencedPO": None, "lineNumber": None,
                "detail": r21["detail"] + " [ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R21",
                "reason": r21["reason"], "issues": r21.get("issues", []),
                "detail": r21["detail"],
            })
        elif existing_r21.get("status") != "FAIL":
            llm_said = existing_r21.get("status")
            existing_r21["status"] = "FAIL"
            existing_r21["detail"] = r21["detail"] + " [ENFORCER OVERRIDE]"
            enforcer_log.append({
                "type": "ARITHMETIC_OVERRIDE", "rule_id": "R21",
                "llm_said": llm_said, "python_says": "FAIL",
                "reason": r21["reason"], "detail": r21["detail"],
            })

    # 5e. Multi-PO Consolidation (R17) — cross-PO item contamination.
    # An item's PO membership is the portion it sits in, so a moved item is
    # only detectable against the PO baselines. The LLM owns R17 too; we
    # inject FAIL when it missed it and override a non-FAIL verdict, mirroring
    # the R20/R21 injection-or-override pattern.
    r17 = _classify_R17(po_baselines, asn_parsed)
    existing_r17 = next((r for r in master_report.get("ruleResults", [])
                        if r.get("rule_id") == "R17"), None)
    if r17["status"] == "FAIL":
        if existing_r17 is None:
            master_report["ruleResults"].append({
                "rule_id": "R17", "name": "Multi-PO Consolidation Check",
                "status": "FAIL", "severity": _RULE_SEVERITY["R17"],
                "referencedPO": r17.get("referencedPO"),
                "lineNumber": r17.get("lineNumber"),
                "detail": r17["detail"] + " [ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R17",
                "reason": r17["reason"], "detail": r17["detail"],
            })
        elif existing_r17.get("status") != "FAIL":
            llm_said = existing_r17.get("status")
            existing_r17["status"] = "FAIL"
            existing_r17["detail"] = r17["detail"] + " [ENFORCER OVERRIDE]"
            enforcer_log.append({
                "type": "ARITHMETIC_OVERRIDE", "rule_id": "R17",
                "llm_said": llm_said, "python_says": "FAIL",
                "reason": r17["reason"], "detail": r17["detail"],
            })

    # 6. ShipmentID Length Check (R19) — Ariba hard limit: > 35 chars causes silent drop
    r19 = _classify_R19(shipment_id, max_length=35)
    if r19["status"] == "FAIL":
        r19_found = any(r.get("rule_id") == "R19" and r["status"] == "FAIL"
                        for r in master_report.get("ruleResults", []))
        if not r19_found:
            master_report["ruleResults"].append({
                "rule_id": "R19", "name": "ShipmentID Length",
                "status": "FAIL", "severity": "CRITICAL",
                "referencedPO": None, "lineNumber": None,
                "detail": (f"shipmentID '{shipment_id}' is {r19['length']} characters. "
                           f"Ariba SCC limit is 35 characters. ASN will be silently dropped."
                           f" [ENFORCER INJECTED]"),
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED", "rule_id": "R19",
                "detail": f"shipmentID length {r19['length']} > 35",
            })

    validated_report = {
        **master_report,
        "enforcer_log": enforcer_log,
        "enforcer_summary": {
            "overrides": sum(1 for e in enforcer_log if e["type"] == "ARITHMETIC_OVERRIDE"),
            "schema_fixes": sum(1 for e in enforcer_log if e["type"] == "SCHEMA_ERROR"),
            "injected_rules": sum(1 for e in enforcer_log
                                   if e["type"] in ("MISSING_RULE_INJECTED",
                                                    "R16_PARTIAL_SHIPMENT_INJECTED")),
            "partial_shipments": sum(1 for e in enforcer_log
                                     if e["type"] in ("UNDER_SHIP_ROUTED_TO_R16",
                                                      "R16_PARTIAL_SHIPMENT_INJECTED",
                                                      "R16_PARTIAL_SHIPMENT_ESCALATED")),
            "partial_deliveries": sum(1 for e in enforcer_log
                                      if e["type"] == "R06_PARTIAL_DELIVERY_DOWNGRADED"),
            "clean": len(enforcer_log) == 0,
            "mode": "deterministic",
        },
    }
    return validated_report


# ══════════════════════════════════════════════════════════════
# Stage 6 (ablation) — No-op enforcer
# ══════════════════════════════════════════════════════════════

def noop_enforce_output(master_report: dict, po_baselines: dict, asn_parsed: dict) -> dict:
    """Pass-through enforcer for ablation: trust the LLM verdicts as-is.
    Provides the same output contract (enforcer_log + enforcer_summary) so
    Stage 7 and downstream consumers work unchanged.

    Mode A is the unprocessed-LLM baseline by design: no overrides, no
    injections, and -- explicitly -- no severity capping.  If the LLM emits
    a severity that does not match the rule register, that mis-emission must
    surface in disposition outcomes so the ablation reflects what an
    unaugmented LLM produces.  Severity capping is what modes B/C/D/E add on
    top of this baseline; including it here would defeat the comparison.
    """
    return {
        **master_report,
        "enforcer_log": [],
        "enforcer_summary": {
            "overrides": 0,
            "schema_fixes": 0,
            "injected_rules": 0,
            "clean": True,
            "mode": "none",
        },
    }


# ══════════════════════════════════════════════════════════════
# Stage 6 (ablation) — LLM-as-enforcer
# ══════════════════════════════════════════════════════════════

_LLM_ENFORCER_SYSTEM = """You are a senior ASN validation auditor. A first-pass
LLM has produced a validation report for a cXML ASN against one or more PO
baselines. Your job is to audit that report and emit only corrections.

You may:
  - Override any ruleResult.status (PASS<->FAIL) if the arithmetic/logic of the
    PO vs ASN clearly contradicts the first pass.
  - Inject a missing FAIL for R18 (duplicate ASN, flagged below) or R19
    (shipmentID > 35 characters) when the condition is present but not caught.
  - Cap a semantic finding severity (date_plausibility and description_coherence
    must not exceed WARNING).

Output STRICT JSON with a single key "overrides". Each override is one of:
  {"action": "override_rule", "rule_id": "<Rxx>", "referencedPO": "<po|null>",
   "lineNumber": <int|null>, "new_status": "PASS|FAIL",
   "reason": "<one sentence>"}
  {"action": "inject_rule", "rule_id": "R18|R19",
   "severity": "CRITICAL", "detail": "<one sentence>"}
  {"action": "cap_semantic_severity", "category": "<category>",
   "new_severity": "WARNING|INFO",
   "reason": "<one sentence>"}

If the first-pass report is correct, return {"overrides": []}. Do not invent
new rules. Do not echo the report back."""

# JSON schema for the enforcer's output, compatible with Azure OpenAI strict
# structured outputs. Uses a flat object with all fields required (nullable
# where not applicable to a given action) because strict mode disallows
# oneOf and requires every property in `required`.
_ENFORCER_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["overrides"],
    "properties": {
        "overrides": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "action", "rule_id", "referencedPO", "lineNumber",
                    "new_status", "severity", "detail", "category",
                    "new_severity", "reason",
                ],
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["override_rule", "inject_rule", "cap_semantic_severity"],
                    },
                    "rule_id": {"type": ["string", "null"]},
                    "referencedPO": {"type": ["string", "null"]},
                    "lineNumber": {"type": ["integer", "null"]},
                    "new_status": {
                        "type": ["string", "null"],
                        "enum": ["PASS", "FAIL", None],
                    },
                    "severity": {
                        "type": ["string", "null"],
                        "enum": ["CRITICAL", "WARNING", "INFO", None],
                    },
                    "detail": {"type": ["string", "null"]},
                    "category": {"type": ["string", "null"]},
                    "new_severity": {
                        "type": ["string", "null"],
                        "enum": ["WARNING", "INFO", None],
                    },
                    "reason": {"type": ["string", "null"]},
                },
            },
        },
    },
}


def llm_enforce_output(master_report: dict, po_baselines: dict, asn_parsed: dict,
                        use_schema: bool = False) -> dict:
    """LLM-as-enforcer: a second LLM call reviews the first LLM's verdicts.

    Same output contract as `enforce_output` so Stage 7 works unchanged. The
    model sees the PO baselines, the parsed ASN and the first-pass report, and
    is asked to emit only corrections (overrides + injections). The auditor
    reviews all 22 rules; no scope filter is applied.

    use_schema=True uses Azure structured outputs (json_schema mode) with
    server-side schema enforcement. Kept off the principal "llm" ablation mode
    so the LLM-vs-deterministic substitution test is not confounded with
    structured outputs."""
    enforcer_log = []

    # Ensure required keys exist (same schema guard as deterministic enforcer)
    for key, default in (("alignment", {"portions": []}),
                         ("ruleResults", []),
                         ("semanticFindings", []),
                         ("overallAssessment", "N/A")):
        if key not in master_report:
            master_report[key] = default
            enforcer_log.append({"type": "SCHEMA_ERROR", "detail": f"Missing key: {key}"})

    # Precompute R18 duplicate flag + R19 length so the auditor LLM has the
    # facts, not a guess. Do not inject ourselves -- we want the LLM to decide.
    shipment_id = _r18_shipment_id(asn_parsed)
    ref_pos = tuple(sorted(
        p.get("referencedPO", "") for p in asn_parsed.get("portions", [])
    ))
    asn_key = (shipment_id, ref_pos)
    is_duplicate = (asn_key in _seen_asns) and not DISABLE_R18
    shipment_id_len = len(shipment_id)

    user_payload = {
        "is_duplicate_asn": is_duplicate,
        "shipmentID": shipment_id,
        "shipmentID_length": shipment_id_len,
        "ariba_limit": 35,
        "po_baselines": {
            pid: {"orderID": bl["orderID"], "currency": bl["currency"],
                  "total": bl["total"],
                  "lineItems": [{"lineNumber": li["lineNumber"],
                                 "quantity": li["quantity"],
                                 "unitPrice": li["unitPrice"],
                                 "uom": li.get("uom", ""),
                                 "requestedDeliveryDate": li.get("requestedDeliveryDate", ""),
                                 "tolerances": li.get("tolerances", {})}
                                for li in bl["lineItems"]]}
            for pid, bl in po_baselines.items()
        },
        "asn_parsed": {
            "shipmentID": asn_parsed.get("shipmentID", ""),
            "deliveryDate": asn_parsed.get("deliveryDate", ""),
            "portions": [
                {"referencedPO": p.get("referencedPO", ""),
                 "items": [{"lineNumber": it["lineNumber"],
                            "quantity": it.get("quantity"),
                            "unitPrice": it.get("unitPrice"),
                            "uom": it.get("uom", "")}
                           for it in p.get("items", [])]}
                for p in asn_parsed.get("portions", [])
            ],
        },
        "first_pass_report": {
            "ruleResults": master_report.get("ruleResults", []),
            "semanticFindings": master_report.get("semanticFindings", []),
        },
    }

    system_prompt = _LLM_ENFORCER_SYSTEM
    schema = _ENFORCER_OUTPUT_SCHEMA if use_schema else None
    raw = llm_call(system_prompt, json.dumps(user_payload), temperature=0.0,
                   schema=schema, schema_name="enforcer_overrides")
    try:
        parsed = json.loads(raw)
        overrides = parsed.get("overrides", [])
    except Exception as e:
        print(f"  [LLM-ENFORCER PARSE ERROR] {type(e).__name__}: {e}")
        overrides = []
        enforcer_log.append({"type": "SCHEMA_ERROR",
                             "detail": f"LLM-enforcer JSON parse failed: {e}"})

    rule_results = master_report.get("ruleResults", [])
    sem_findings = master_report.get("semanticFindings", [])

    for ov in overrides:
        action = ov.get("action", "")
        if action == "override_rule":
            rid = ov.get("rule_id", "")
            ref_po = ov.get("referencedPO")
            line_num = ov.get("lineNumber")
            new_status = ov.get("new_status", "")
            if new_status not in ("PASS", "FAIL"):
                continue
            for r in rule_results:
                if (r.get("rule_id") == rid
                        and r.get("referencedPO") == ref_po
                        and r.get("lineNumber") == line_num):
                    if r.get("status") != new_status:
                        enforcer_log.append({
                            "type": "ARITHMETIC_OVERRIDE",
                            "rule_id": rid,
                            "referencedPO": ref_po,
                            "lineNumber": line_num,
                            "llm_said": r.get("status"),
                            "python_says": new_status,
                            "detail": ov.get("reason", ""),
                        })
                        r["status"] = new_status
                        r["detail"] = (r.get("detail", "")
                                       + f" [LLM-ENFORCER OVERRIDE -> {new_status}]")
                    break
        elif action == "inject_rule":
            rid = ov.get("rule_id", "")
            if rid not in ("R18", "R19"):
                continue
            already = any(r.get("rule_id") == rid and r.get("status") == "FAIL"
                          for r in rule_results)
            if already:
                continue
            rule_results.append({
                "rule_id": rid,
                "name": "Duplicate ASN Detection" if rid == "R18" else "ShipmentID Length",
                "status": "FAIL",
                "severity": ov.get("severity", "CRITICAL"),
                "referencedPO": None,
                "lineNumber": None,
                "detail": ov.get("detail", "") + " [LLM-ENFORCER INJECTED]",
            })
            enforcer_log.append({
                "type": "MISSING_RULE_INJECTED",
                "rule_id": rid,
                "detail": ov.get("detail", ""),
            })
        elif action == "cap_semantic_severity":
            category = ov.get("category", "")
            new_sev = ov.get("new_severity", "")
            if new_sev not in ("WARNING", "INFO"):
                continue
            _rank = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
            for f in sem_findings:
                if (f.get("category") == category
                        and _rank.get(f.get("severity", "INFO"), 0) > _rank[new_sev]):
                    enforcer_log.append({
                        "type": "SEMANTIC_SEVERITY_CAPPED",
                        "category": category,
                        "llm_said": f.get("severity"),
                        "capped_to": new_sev,
                        "detail": ov.get("reason", ""),
                    })
                    f["severity"] = new_sev

    # Still record seen ASN so subsequent runs can flag duplicates, regardless
    # of whether the LLM-enforcer chose to inject R18 for this one.
    if not is_duplicate:
        _seen_asns.add(asn_key)
        _save_seen_asns(_seen_asns)

    # Severity capping over all ruleResults (first-pass + auditor-injected).
    # Runs last so caps cover whatever the auditor produced as well.
    # po_baselines/asn_parsed enable the conditional R06 severity (partial
    # delivery -> WARNING) so this mode no longer over-rejects partials.
    n_caps = _cap_rule_severities(rule_results, enforcer_log,
                                  po_baselines=po_baselines, asn_parsed=asn_parsed)

    suffix = "_schema" if use_schema else ""

    return {
        **master_report,
        "enforcer_log": enforcer_log,
        "enforcer_summary": {
            "overrides": sum(1 for e in enforcer_log if e["type"] == "ARITHMETIC_OVERRIDE"),
            "schema_fixes": sum(1 for e in enforcer_log if e["type"] == "SCHEMA_ERROR"),
            "injected_rules": sum(1 for e in enforcer_log if e["type"] == "MISSING_RULE_INJECTED"),
            "severity_caps": n_caps,
            "clean": len(enforcer_log) == 0,
            "mode": f"llm{suffix}",
            "use_schema": use_schema,
        },
    }


def tool_use_enforce_output(master_report: dict, po_baselines: dict, asn_parsed: dict) -> dict:
    """Pass-through Stage 6 for tool_use mode, with one exception: R18.

    Stage 5 was invoked with use_tools=True so the model already exercised
    in-generation deterministic verification for R01/R02/R03/R06/R15/R16/R19.
    R18 (Duplicate ASN Detection) is structurally not expressible as a
    stateless tool call -- it requires the persistent set of previously-seen
    (shipmentID, referencedPO-set) keys. We mirror the deterministic
    enforcer's R18 logic here as a post-hoc fallback so mode D is not blind
    to duplicates. The fallback is logged distinctly from Stage 5 tool calls."""
    enforcer_log = []
    rule_results = master_report.setdefault("ruleResults", [])

    shipment_id = _r18_shipment_id(asn_parsed)
    ref_pos = tuple(sorted(
        p.get("referencedPO", "") for p in asn_parsed.get("portions", [])
    ))
    asn_key = (shipment_id, ref_pos)

    r18_injected = False
    if (asn_key in _seen_asns) and not DISABLE_R18:
        already_failed = any(r.get("rule_id") == "R18" and r.get("status") == "FAIL"
                             for r in rule_results)
        if not already_failed:
            rule_results.append({
                "rule_id": "R18", "name": "Duplicate ASN Detection",
                "status": "FAIL", "severity": _RULE_SEVERITY["R18"],
                "referencedPO": None, "lineNumber": None,
                "detail": (f"Duplicate ASN: shipmentID={shipment_id} "
                           f"[TOOL_USE R18 POST-HOC FALLBACK]"),
            })
            enforcer_log.append({
                "stage": 6,
                "type": "R18_POSTHOC_FALLBACK",
                "rule_id": "R18",
                "detail": f"Duplicate ASN {shipment_id} for POs {list(ref_pos)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            r18_injected = True
    else:
        _seen_asns.add(asn_key)
        _save_seen_asns(_seen_asns)

    # Severity capping over the LLM's tool-use emissions (and the post-hoc
    # R18 we may have just injected). po_baselines/asn_parsed enable the
    # conditional R06 severity (partial delivery -> WARNING).
    n_caps = _cap_rule_severities(rule_results, enforcer_log,
                                  po_baselines=po_baselines, asn_parsed=asn_parsed)

    return {
        **master_report,
        "ruleResults": rule_results,
        "enforcer_log": enforcer_log,
        "enforcer_summary": {
            "overrides": 0,
            "schema_fixes": 0,
            "injected_rules": 1 if r18_injected else 0,
            "severity_caps": n_caps,
            "clean": (not r18_injected) and (n_caps == 0),
            "mode": "tool_use",
            "tool_calls": master_report.get("tool_calls_log", []),
            "r18_posthoc_fallback": r18_injected,
        },
    }


def _dispatch_enforcer(mode: str, master_report: dict, po_baselines: dict, asn_parsed: dict) -> dict:
    """Route Stage 6 to the selected enforcer implementation."""
    if mode == "none":
        return noop_enforce_output(master_report, po_baselines, asn_parsed)
    if mode == "llm":
        return llm_enforce_output(master_report, po_baselines, asn_parsed,
                                    use_schema=False)
    if mode == "tool_use":
        return tool_use_enforce_output(master_report, po_baselines, asn_parsed)
    if mode == "partition":
        # The merge already happened in partition_validate_and_enforce; the
        # master_report passed in IS the validated report.
        return master_report
    # default + "deterministic"
    return enforce_output(master_report, po_baselines, asn_parsed)


# ══════════════════════════════════════════════════════════════
# Mode E — Partition (Stage 5a deterministic + Stage 5b LLM-narrow)
# ══════════════════════════════════════════════════════════════

# The 14 rules NOT in _DETERMINISTIC_OWNED_RULES.  Mode E asks the LLM to
# evaluate only these.
_NON_OWNED_RULES = tuple(r["id"] for r in RULE_DEFINITIONS
                          if r["id"] not in _DETERMINISTIC_OWNED_RULES)


def compute_owned_rule_verdicts(po_baselines: dict, asn_parsed: dict,
                                  seen_asns: set | None = None) -> list[dict]:
    """Stage 5a — deterministic verdicts for the 8 owned rules.

    Returns exactly 8 ruleResult dicts, one per rule in _DETERMINISTIC_OWNED_RULES,
    in numeric order.  Per-(PO, line) checks are aggregated to a single verdict
    per rule (status = worst per-line status; severity per RULE_DEFINITIONS;
    detail rolls up failing lines).  Mutates `seen_asns` for R18 if not duplicate.

    Formulas mirror enforce_output()'s arithmetic blocks via the shared
    `_classify_R0x` helpers above.
    """
    if seen_asns is None:
        seen_asns = _seen_asns

    # Build per-line lookups (same as enforce_output).
    po_line_lookup = {(oid, li["lineNumber"]): li
                      for oid, bl in po_baselines.items()
                      for li in bl["lineItems"]}
    asn_line_lookup = {}
    for portion in asn_parsed.get("portions", []):
        po_id = portion.get("referencedPO", "")
        for item in portion.get("items", []):
            asn_line_lookup[(po_id, item["lineNumber"])] = item

    # Severity per owned rule (from RULE_DEFINITIONS).
    sev_default = {r["id"]: r["severity"] for r in RULE_DEFINITIONS}

    # Per-line collectors for aggregation.
    r01_lines, r02_lines, r03_lines, r15_lines = [], [], [], []
    r01_under_ship_lines = []  # feeds R16

    for (po_id, line_num), asn_line in asn_line_lookup.items():
        po_line = po_line_lookup.get((po_id, line_num))
        if not po_line:
            continue
        tol = po_line.get("tolerances", {})

        # R01
        r01 = _classify_R01(po_line["quantity"], asn_line["quantity"],
                             tol.get("quantity_lower_pct", 0),
                             tol.get("quantity_upper_pct", 0))
        r01_lines.append({"po": po_id, "line": line_num, **r01,
                          "po_qty": po_line["quantity"],
                          "asn_qty": asn_line["quantity"]})
        if r01["ship_case"] == "under_ship":
            r01_under_ship_lines.append({
                "po": po_id, "line": line_num,
                "po_qty": po_line["quantity"], "asn_qty": asn_line["quantity"],
                "remaining": po_line["quantity"] - asn_line["quantity"],
            })

        # R02
        r02 = _classify_R02(po_line["unitPrice"], asn_line["unitPrice"],
                             tol.get("price_lower_pct", 0),
                             tol.get("price_upper_pct", 0))
        r02_lines.append({"po": po_id, "line": line_num, **r02,
                          "po_price": po_line["unitPrice"],
                          "asn_price": asn_line["unitPrice"]})

        # R15
        r15 = _classify_R15(po_line["quantity"], po_line["unitPrice"],
                             asn_line["quantity"], asn_line["unitPrice"])
        r15_lines.append({"po": po_id, "line": line_num, **r15})

        # R03 (uses ASN-header deliveryDate per portion's PO line)
        r03 = _classify_R03(po_line.get("requestedDeliveryDate", ""),
                             asn_parsed.get("deliveryDate", ""),
                             tol.get("time_lower_days", 0),
                             tol.get("time_upper_days", 0))
        if r03["status"] is not None:
            r03_lines.append({"po": po_id, "line": line_num, **r03})

    # ---- Aggregate per-rule verdicts ----

    def _agg_simple(rule_id: str, name: str, per_line: list,
                    fail_severity: str, pass_severity: str) -> dict:
        fails = [x for x in per_line if x["status"] == "FAIL"]
        if not per_line:
            return {"rule_id": rule_id, "name": name, "status": "PASS",
                    "severity": pass_severity, "referencedPO": None,
                    "lineNumber": None,
                    "detail": f"{rule_id}: no lines to evaluate"}
        if not fails:
            return {"rule_id": rule_id, "name": name, "status": "PASS",
                    "severity": pass_severity, "referencedPO": None,
                    "lineNumber": None,
                    "detail": f"{rule_id}: {len(per_line)} line(s) within tolerance"}
        return {"rule_id": rule_id, "name": name, "status": "FAIL",
                "severity": fail_severity,
                "referencedPO": fails[0]["po"], "lineNumber": fails[0]["line"],
                "detail": (f"{rule_id}: {len(fails)}/{len(per_line)} line(s) failed; "
                           f"first: PO {fails[0]['po']} line {fails[0]['line']}")}

    name_of = {r["id"]: r["name"] for r in RULE_DEFINITIONS}

    # R01 — over_ship is the only FAIL; under_ship is PASS (queues R16).
    over_ships = [x for x in r01_lines if x["ship_case"] == "over_ship"]
    if not r01_lines:
        v_r01 = {"rule_id": "R01", "name": name_of["R01"], "status": "PASS",
                 "severity": sev_default["R01"], "referencedPO": None,
                 "lineNumber": None, "detail": "R01: no lines to evaluate"}
    elif over_ships:
        f = over_ships[0]
        v_r01 = {"rule_id": "R01", "name": name_of["R01"], "status": "FAIL",
                 "severity": sev_default["R01"],
                 "referencedPO": f["po"], "lineNumber": f["line"],
                 "detail": (f"R01: {len(over_ships)}/{len(r01_lines)} line(s) over-ship; "
                            f"first: PO {f['po']} line {f['line']} "
                            f"qty {f['asn_qty']} > upper {f['upper_bound']}")}
    else:
        unders = len(r01_under_ship_lines)
        v_r01 = {"rule_id": "R01", "name": name_of["R01"], "status": "PASS",
                 "severity": sev_default["R01"], "referencedPO": None,
                 "lineNumber": None,
                 "detail": (f"R01: {len(r01_lines)} line(s) within band"
                            + (f"; {unders} under-ship routed to R16" if unders else ""))}

    v_r02 = _agg_simple("R02", name_of["R02"], r02_lines,
                         sev_default["R02"], sev_default["R02"])

    v_r03 = _agg_simple("R03", name_of["R03"], r03_lines,
                         sev_default["R03"], sev_default["R03"])

    v_r15 = _agg_simple("R15", name_of["R15"], r15_lines,
                         sev_default["R15"], sev_default["R15"])

    # R06 — delegate to shared _classify_R06 (same logic used by mode B).
    r06_raw = _classify_R06(po_baselines, asn_parsed)
    v_r06 = {
        "rule_id": "R06", "name": name_of["R06"],
        "status": r06_raw["status"], "severity": r06_raw["severity"],
        "referencedPO": r06_raw["referencedPO"],
        "lineNumber": r06_raw["lineNumber"],
        "detail": r06_raw["detail"],
    }

    # R16 — escalated when any R01 line is under_ship; PASS/INFO otherwise.
    if r01_under_ship_lines:
        u = r01_under_ship_lines[0]
        v_r16 = {"rule_id": "R16", "name": name_of["R16"], "status": "FAIL",
                 "severity": "WARNING",
                 "referencedPO": u["po"], "lineNumber": u["line"],
                 "detail": (f"R16: {len(r01_under_ship_lines)} line(s) under-shipped; "
                            f"first: PO {u['po']} line {u['line']} "
                            f"qty {u['asn_qty']}/{u['po_qty']} "
                            f"(remaining {u['remaining']}) "
                            f"[ENFORCER ESCALATED -> WARNING]")}
    else:
        v_r16 = {"rule_id": "R16", "name": name_of["R16"], "status": "PASS",
                 "severity": "INFO",
                 "referencedPO": None, "lineNumber": None,
                 "detail": "R16: no partial shipment detected"}

    # R18 — header-level, stateful.
    shipment_id = _r18_shipment_id(asn_parsed)
    ref_pos = tuple(sorted(
        p.get("referencedPO", "") for p in asn_parsed.get("portions", [])
    ))
    r18 = _classify_R18(shipment_id, ref_pos, seen_asns, mutate=True)
    v_r18 = {"rule_id": "R18", "name": name_of["R18"],
             "status": r18["status"], "severity": sev_default["R18"],
             "referencedPO": None, "lineNumber": None,
             "detail": (f"R18: shipmentID '{shipment_id}' for POs {list(ref_pos)} "
                        + ("seen previously [DUPLICATE]"
                           if r18["is_duplicate"] else "first sighting"))}

    # R19 — header-level.
    r19 = _classify_R19(shipment_id, max_length=35)
    v_r19 = {"rule_id": "R19", "name": name_of["R19"],
             "status": r19["status"], "severity": sev_default["R19"],
             "referencedPO": None, "lineNumber": None,
             "detail": (f"R19: shipmentID length {r19['length']} "
                        + ("> 35 [Ariba SCC silent drop]"
                           if r19["status"] == "FAIL" else "<= 35"))}

    return [v_r01, v_r02, v_r03, v_r06, v_r15, v_r16, v_r18, v_r19]


def build_narrow_validation_prompt(po_baselines: dict, asn_parsed: dict,
                                     deterministic_verdicts: list[dict]) -> dict:
    """Stage 5b — slimmed prompt covering only the 14 non-owned rules.

    The 8 deterministic verdicts from Stage 5a are embedded as input context
    so the LLM sees them but cannot re-emit them.  Output JSON shape mirrors
    build_validation_prompt() so llm_validate() can be reused as-is."""
    rules_text = "\n".join(
        f"  - {r['id']} ({r['severity']}): {r['name']} -- {r['description']}"
        for r in RULE_DEFINITIONS if r["id"] in _NON_OWNED_RULES
    )

    owned_list = ", ".join(_DETERMINISTIC_OWNED_RULES)
    non_owned_list = ", ".join(_NON_OWNED_RULES)

    system_prompt = f"""You are an expert procurement validation agent for Hilti.
A deterministic preprocessor has already evaluated the eight rules
{owned_list} for this ASN.  Your job is to evaluate ONLY the remaining
fourteen rules ({non_owned_list}).

DO NOT emit ruleResults for {owned_list}.  Their verdicts are fixed and
provided to you as read-only input under "deterministic_verdicts".

You must still:
  1. Produce a LINE ALIGNMENT block matching ASN items to PO lines using
     (lineNumber, buyerPartID).
  2. Produce ruleResults entries for each of the fourteen non-owned rules
     ({non_owned_list}).  One entry per rule -- aggregate per-line failures
     into a single verdict (status = worst per-line status, severity per
     rule definition, detail rolls up failing lines).
  3. Produce semanticFindings as usual (description coherence, date /
     packaging plausibility, cross-PO patterns).
  4. Produce overallAssessment: a one- to two-sentence summary.

NON-OWNED RULES TO EVALUATE:
{rules_text}

Return ONLY valid JSON with this exact structure:
{{
  "alignment": {{ "portions": [...] }},
  "ruleResults": [
    {{
      "rule_id": "<one of {non_owned_list}>",
      "name": "<rule name>",
      "status": "<PASS|FAIL>",
      "severity": "<CRITICAL|WARNING|INFO>",
      "referencedPO": "<orderID or null>",
      "lineNumber": <int or null>,
      "detail": "<explanation>"
    }}
  ],
  "semanticFindings": [...],
  "overallAssessment": "<brief summary>"
}}

Hard constraints:
  - Exactly fourteen ruleResults entries, covering each of {non_owned_list}
    once and only once.
  - DO NOT emit ruleResults for {owned_list} -- they will be rejected.
"""

    # Supplier-conditioned prompting (RAG): inject the same one-line historical
    # note used by build_validation_prompt(), so the pre-hoc (partition / Mode E)
    # arm actually exercises the supplier prior on the non-owned rules it
    # evaluates (R13, R14, R21, ...). Previously this injection lived only in the
    # full-prompt path, so the partition arm logged 0 injected lines.
    supplier_profile = _load_supplier_profile(asn_parsed, po_baselines)
    if supplier_profile and supplier_profile.get("prompt_injection_line"):
        system_prompt = (
            f"SUPPLIER HISTORY ({supplier_profile['vendor_id']}): "
            f"{supplier_profile['prompt_injection_line']}\n\n"
            f"{system_prompt}"
        )

    po_data = {oid: {"orderID": bl["orderID"],
                      "orderDate": bl.get("orderDate", ""),
                      "currency": bl["currency"], "total": bl["total"],
                      "incoterms": bl.get("incoterms", ""),
                      "shipTo": bl.get("shipTo", {}),
                      "supplier": bl.get("supplier", {}),
                      "lineItems": bl["lineItems"]}
                for oid, bl in po_baselines.items()}

    user_payload = {
        "po_baselines": po_data,
        "asn_parsed": asn_parsed,
        "deterministic_verdicts (already computed, do not re-evaluate)":
            deterministic_verdicts,
    }
    user_prompt = json.dumps(user_payload, indent=2, ensure_ascii=False)

    context_metadata = {
        "shipmentID": asn_parsed.get("shipmentID", ""),
        "po_count": len(po_baselines),
        "po_orderIDs": list(po_baselines.keys()),
        "asn_portions": len(asn_parsed.get("portions", [])),
        "asn_total_items": sum(len(p.get("items", []))
                                 for p in asn_parsed.get("portions", [])),
        "rules_injected": len(_NON_OWNED_RULES),
        "prompt_system_chars": len(system_prompt),
        "prompt_user_chars": len(user_prompt),
        "few_shot_enabled": False,
        "stage5_schema_enabled": False,
        "narrow_mode": True,
        "supplier_profile_used": supplier_profile["vendor_id"] if supplier_profile else None,
        "supplier_profile_n_prior": supplier_profile.get("n_prior_asns") if supplier_profile else None,
        "rag_placebo": bool(supplier_profile.get("placebo")) if supplier_profile else False,
        "rag_placebo_source": supplier_profile.get("placebo_source") if supplier_profile else None,
    }
    return {"system_prompt": system_prompt, "user_prompt": user_prompt,
            "context_metadata": context_metadata}


def partition_validate_and_enforce(po_baselines: dict, asn_parsed: dict) -> dict:
    """Mode E orchestrator: Stage 5a (deterministic) + Stage 5b (LLM-narrow) + merge.

    Returns a master_report in the same shape as enforce_output()'s output, so
    Stage 7 (Decision Router) consumes it without changes.  By construction:
      - ruleResults has exactly 22 entries, one per rule R01..R22 in numeric order.
      - enforcer_summary.mode = "partition", overrides=0, injected_rules=0.
      - enforcer_log carries 8 PARTITION_DETERMINISTIC entries (one per owned
        rule) plus PARTITION_LLM_DROPPED entries for any owned-rule emissions
        the LLM made anyway.
    """
    enforcer_log = []

    # Stage 5a
    deterministic_verdicts = compute_owned_rule_verdicts(
        po_baselines, asn_parsed, _seen_asns
    )
    for v in deterministic_verdicts:
        enforcer_log.append({
            "stage": 5,
            "type": "PARTITION_DETERMINISTIC",
            "rule_id": v["rule_id"],
            "status": v["status"],
            "severity": v["severity"],
            "detail": v["detail"],
        })

    # Stage 5b
    narrow_prompt = build_narrow_validation_prompt(
        po_baselines, asn_parsed, deterministic_verdicts
    )
    print(f"  [Stage 5a] Deterministic verdicts: {len(deterministic_verdicts)}")
    print(f"  [Stage 5b] Narrow prompt: "
          f"{narrow_prompt['context_metadata']['prompt_system_chars']
              + narrow_prompt['context_metadata']['prompt_user_chars']:,} chars "
          f"({narrow_prompt['context_metadata']['rules_injected']} rules)")
    llm_report = llm_validate(narrow_prompt, use_tools=False)

    # Filter LLM output to non-owned rules only; drop owned-rule emissions.
    llm_rule_results = llm_report.get("ruleResults", [])
    kept = []
    seen_non_owned = set()
    for r in llm_rule_results:
        rid = r.get("rule_id", "")
        if rid in _DETERMINISTIC_OWNED_RULES:
            enforcer_log.append({
                "stage": 5,
                "type": "PARTITION_LLM_DROPPED",
                "rule_id": rid,
                "detail": (f"LLM emitted owned-rule {rid} despite instruction; "
                           f"dropped (deterministic verdict is authoritative)"),
            })
            continue
        if rid not in _NON_OWNED_RULES:
            enforcer_log.append({
                "stage": 5,
                "type": "PARTITION_LLM_DROPPED",
                "rule_id": rid,
                "detail": f"LLM emitted unknown rule_id '{rid}'; dropped",
            })
            continue
        if rid in seen_non_owned:
            enforcer_log.append({
                "stage": 5,
                "type": "PARTITION_LLM_DROPPED",
                "rule_id": rid,
                "detail": f"LLM emitted duplicate {rid}; kept first occurrence only",
            })
            continue
        seen_non_owned.add(rid)
        kept.append(r)

    # Severity capping over the 14 LLM-emitted non-owned rules.  The 8 owned
    # rules are written by compute_owned_rule_verdicts using RULE_DEFINITIONS
    # severities, so they are correct by construction and excluded from cap.
    n_caps = _cap_rule_severities(kept, enforcer_log, scope=_NON_OWNED_RULES)

    # Stub any non-owned rule the LLM omitted -- emitted as PASS/INFO with a
    # diagnostic note so the 22-entry invariant is unconditional.
    name_of = {r["id"]: r["name"] for r in RULE_DEFINITIONS}
    sev_of = {r["id"]: r["severity"] for r in RULE_DEFINITIONS}
    for rid in _NON_OWNED_RULES:
        if rid in seen_non_owned:
            continue
        kept.append({
            "rule_id": rid, "name": name_of[rid],
            "status": "PASS", "severity": sev_of[rid],
            "referencedPO": None, "lineNumber": None,
            "detail": f"{rid}: not emitted by Stage 5b LLM; defaulted to PASS",
        })
        enforcer_log.append({
            "stage": 5,
            "type": "PARTITION_LLM_OMITTED",
            "rule_id": rid,
            "detail": f"LLM did not emit {rid}; default PASS injected",
        })

    # Deterministic override for two non-owned rules whose violations the LLM
    # cannot reliably see on its own: R17 (cross-PO contamination — needs the PO
    # baselines) and R20 (weight plausibility — needs the parsed/raw weights).
    # Mirrors the deterministic enforcer (enforce_output steps 5c/5e) so the
    # partition design closes the same coverage boundary. Rule-FIRING only;
    # severities use the registered values.
    det_overrides = 0
    for rid, verdict in (("R17", _classify_R17(po_baselines, asn_parsed)),
                         ("R20", _classify_R20(asn_parsed))):
        if verdict.get("status") != "FAIL":
            continue
        entry = next((r for r in kept if r.get("rule_id") == rid), None)
        if entry is None or entry.get("status") == "FAIL":
            continue  # absent (shouldn't happen — stubbed above) or LLM already caught it
        entry["status"] = "FAIL"
        entry["severity"] = _RULE_SEVERITY[rid]
        entry["referencedPO"] = verdict.get("referencedPO")
        entry["lineNumber"] = verdict.get("lineNumber")
        entry["detail"] = verdict["detail"] + " [PARTITION DETERMINISTIC OVERRIDE]"
        enforcer_log.append({
            "stage": 6, "type": "PARTITION_DETERMINISTIC_OVERRIDE",
            "rule_id": rid, "reason": verdict.get("reason"),
            "detail": verdict["detail"],
        })
        det_overrides += 1

    # Merge: 8 deterministic + 14 LLM, sorted by rule_id numerically.
    merged = deterministic_verdicts + kept
    merged.sort(key=lambda r: int(r["rule_id"][1:]))

    schema_fixes = sum(1 for e in enforcer_log if e.get("type") == "SCHEMA_ERROR")
    injected_rules = sum(1 for e in enforcer_log
                          if e.get("type") == "PARTITION_LLM_OMITTED")
    llm_dropped = sum(1 for e in enforcer_log
                       if e.get("type") == "PARTITION_LLM_DROPPED")

    master_report = {
        "alignment": llm_report.get("alignment", {"portions": []}),
        "ruleResults": merged,
        "semanticFindings": llm_report.get("semanticFindings", []),
        "overallAssessment": llm_report.get("overallAssessment", ""),
        "enforcer_log": enforcer_log,
        "enforcer_summary": {
            "overrides": det_overrides,
            "schema_fixes": schema_fixes,
            "injected_rules": injected_rules,
            "severity_caps": n_caps,
            "llm_dropped": llm_dropped,
            "clean": (n_caps == 0
                      and injected_rules == 0
                      and llm_dropped == 0
                      and schema_fixes == 0
                      and det_overrides == 0),
            "mode": "partition",
            "deterministic_verdicts": len(deterministic_verdicts),
            "llm_verdicts": len(kept),
            "total_rules": len(merged),
        },
        # Surface the narrow-prompt metadata (incl. supplier_profile_used for the
        # RAG audit) so the partition result's contextMetadata is not empty.
        "_context_metadata": narrow_prompt["context_metadata"],
    }
    return master_report


# ══════════════════════════════════════════════════════════════
# Stage 7 — Decision Router + Audit
# ══════════════════════════════════════════════════════════════

def compute_disposition(validated_report: dict) -> dict:
    rule_results = validated_report.get("ruleResults", [])
    sem_findings = validated_report.get("semanticFindings", [])

    # Detect LLM failure — do not auto-pass if the LLM didn't actually validate
    llm_failed = validated_report.get("_llm_error", False)
    enforcer_summary = validated_report.get("enforcer_summary", {})
    if enforcer_summary.get("schema_fixes", 0) > 0 and len(rule_results) == 0:
        llm_failed = True  # LLM returned garbage — missing ruleResults entirely

    critical_rule_fails = sum(1 for r in rule_results
                              if r["status"] == "FAIL" and r.get("severity") == "CRITICAL")
    warning_rule_fails = sum(1 for r in rule_results
                             if r["status"] == "FAIL" and r.get("severity") == "WARNING")
    sem_findings = [f for f in sem_findings if isinstance(f, dict)]
    sem_critical = sum(1 for f in sem_findings if f.get("severity") == "CRITICAL")
    sem_warning = sum(1 for f in sem_findings if f.get("severity") == "WARNING")

    total_critical = critical_rule_fails + sem_critical
    total_warning = warning_rule_fails + sem_warning

    if llm_failed and total_critical == 0 and total_warning == 0:
        disposition = "ERROR"
        reason = "LLM validation unavailable -- manual review required"
    elif total_critical > 0:
        disposition = "REJECT"
        reason = f"{total_critical} critical issue(s) found"
    elif total_warning > 0:
        disposition = "REVIEW"
        reason = f"{total_warning} warning(s) require buyer review"
    else:
        disposition = "PASS"
        reason = "All checks passed -- ASN can be auto-accepted"

    action_items = []
    for r in rule_results:
        if r["status"] == "FAIL":
            action_items.append({
                "source": "rule", "rule_id": r["rule_id"],
                "severity": r["severity"], "detail": r["detail"],
                "referencedPO": r.get("referencedPO"),
                "lineNumber": r.get("lineNumber"),
            })
    for f in sem_findings:
        if f.get("severity") in ("CRITICAL", "WARNING"):
            action_items.append({
                "source": "semantic", "category": f.get("category", ""),
                "severity": f["severity"], "detail": f.get("explanation", ""),
                "referencedPO": f.get("referencedPO"),
                "lineNumber": f.get("lineNumber"),
            })

    alignment = validated_report.get("alignment", {})
    portions = alignment.get("portions", [])
    total_matched = sum(len(p.get("matchedLines", [])) for p in portions)
    total_unmatched = sum(len(p.get("unmatchedPO", [])) for p in portions)
    total_phantom = sum(len(p.get("phantomASN", [])) for p in portions)

    return {
        "disposition": disposition,
        "reason": reason,
        "summary": {
            "criticalIssues": total_critical,
            "warningIssues": total_warning,
            "rulesTotal": len(rule_results),
            "rulesPass": sum(1 for r in rule_results if r["status"] == "PASS"),
            "rulesFail": sum(1 for r in rule_results if r["status"] == "FAIL"),
            "semanticFindings": len(sem_findings),
            "portions": len(portions),
            "matchedLines": total_matched,
            "unmatchedPO": total_unmatched,
            "phantomASN": total_phantom,
            "enforcerClean": validated_report.get("enforcer_summary", {}).get("clean", True),
            "enforcerOverrides": validated_report.get("enforcer_summary", {}).get("overrides", 0),
        },
        "actionItems": action_items,
    }


def audit_log(asn_parsed: dict, po_baselines: dict, validated_report: dict, disposition: dict):
    shipment_id = asn_parsed.get("shipmentID", "unknown")
    audit_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shipmentID": shipment_id,
        "referencedPOs": list(po_baselines.keys()),
        "disposition": disposition["disposition"],
        "reason": disposition["reason"],
        "summary": disposition["summary"],
        "actionItems": disposition["actionItems"],
        "enforcer_log": validated_report.get("enforcer_log", []),
        "overallAssessment": validated_report.get("overallAssessment", ""),
    }
    out_path = OUTPUT / "reports" / f"ASN_{_safe_filename_id(shipment_id)}_audit.json"
    out_path.write_text(json.dumps(audit_record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


# ══════════════════════════════════════════════════════════════
# Stage 7c — Markdown Report (human-readable, for Supply Manager)
# ══════════════════════════════════════════════════════════════

_DISPOSITION_BANNER = {
    "PASS":   "APPROVED -- ASN can be auto-accepted",
    "REVIEW": "NEEDS REVIEW -- Buyer action required",
    "REJECT": "REJECTED -- Critical issue(s) found",
}

def generate_markdown_report(
    asn_parsed: dict,
    po_baselines: dict,
    validated_report: dict,
    disposition: dict,
) -> pathlib.Path:
    """Generate a human-readable Markdown validation report."""
    shipment_id = asn_parsed.get("shipmentID", "unknown")
    ref_pos = list(po_baselines.keys())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    disp = disposition["disposition"]
    summary = disposition["summary"]
    action_items = disposition["actionItems"]
    rule_results = validated_report.get("ruleResults", [])
    sem_findings = validated_report.get("semanticFindings", [])
    alignment = validated_report.get("alignment", {})
    enforcer_log = validated_report.get("enforcer_log", [])

    # Derive totals
    po_count = len(ref_pos)
    total_value = 0.0
    currency = ""
    for bl in po_baselines.values():
        total_value += bl.get("total", 0.0)
        if not currency:
            currency = bl.get("currency", "")

    supplier_name = asn_parsed.get("supplierName",
                                    asn_parsed.get("shipFrom", {}).get("name", "N/A"))
    ship_to = asn_parsed.get("shipTo", {})
    ship_to_name = ship_to.get("name", "N/A") if isinstance(ship_to, dict) else "N/A"
    ship_to_id = ship_to.get("addressID", "") if isinstance(ship_to, dict) else ""
    incoterms = asn_parsed.get("transportTerms", "N/A")

    lines = []
    w = lines.append

    # ── Title + Disposition ──
    if disp == "PASS":
        w("# PASS — ASN Validated")
    elif disp == "REVIEW":
        w("# REVIEW — Buyer Action Required")
    else:
        w("# REJECT — Critical Issues Found")
    w("")

    # ── Shipment Overview ──
    w("## Shipment Overview")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| **ASN** | {shipment_id} |")
    w(f"| **Supplier** | {supplier_name} |")
    w(f"| **Ship-To** | {ship_to_name}" + (f" ({ship_to_id})" if ship_to_id else "") + " |")
    w(f"| **PO(s)** | {po_count} ({', '.join(ref_pos)}) |")
    w(f"| **Line Items** | {summary.get('matchedLines', 0)} matched |")
    w(f"| **Total Value** | {currency} {total_value:,.2f} |")
    w(f"| **Ship Date** | {asn_parsed.get('shipmentDate', 'N/A')[:10]} |")
    w(f"| **Delivery Date** | {asn_parsed.get('deliveryDate', 'N/A')[:10]} |")
    w(f"| **Incoterms** | {incoterms} |")
    w(f"| **Validated** | {now} |")
    w("")

    # ── Issues (only if disposition is not PASS) ──
    failed_rules = [r for r in rule_results if r["status"] == "FAIL"]
    sem_findings = [f for f in sem_findings if isinstance(f, dict)]
    warning_semantics = [f for f in sem_findings if f.get("severity") in ("CRITICAL", "WARNING")]

    if failed_rules or warning_semantics:
        w("## Issues")
        w("")

        # Deduplicate failed rules: group by rule_id, show one entry per rule
        from collections import OrderedDict as _OD
        grouped_fails = _OD()
        for r in sorted(failed_rules, key=lambda x: (
            0 if x.get("severity") == "CRITICAL" else 1, x.get("rule_id", ""))):
            rid = r.get("rule_id", "")
            grouped_fails.setdefault(rid, []).append(r)

        for rid, items in grouped_fails.items():
            first = items[0]
            sev = first.get("severity", "")
            name = first.get("name", "")
            detail = first.get("detail", "").replace("|", "/").replace("\n", " ").strip()
            first_sent = detail.split(". ")[0].rstrip(".") + "." if detail else ""
            if len(first_sent) > 200:
                first_sent = first_sent[:197] + "..."

            w(f"**{rid} {name}** — {sev}")
            w(f": {first_sent}")

            # List affected POs compactly
            affected = []
            for r in items:
                ref_po = r.get("referencedPO", "")
                if ref_po and ref_po not in affected:
                    affected.append(ref_po)
            if len(affected) == 1:
                w(f": PO {affected[0]}")
            elif affected:
                w(f": Affected POs: {', '.join(affected)}")

            # Show one enforcer override summary (not per-line)
            if enforcer_log:
                relevant = [e for e in enforcer_log if e.get("rule_id") == rid]
                if relevant:
                    override_detail = relevant[0].get("detail", "")
                    suffix = f" (+{len(relevant)-1} more)" if len(relevant) > 1 else ""
                    w(f": *Enforcer override: {override_detail}{suffix}*")
            w("")

        # Semantic warnings (deduplicate by category)
        seen_cats = set()
        for f in warning_semantics:
            cat = f.get("category", "General")
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            sev = f.get("severity", "WARNING")
            expl = f.get("explanation", "").replace("|", "/").replace("\n", " ").strip()
            first_sent = expl.split(". ")[0].rstrip(".") + "." if expl else ""
            if len(first_sent) > 200:
                first_sent = first_sent[:197] + "..."
            w(f"**{cat}** — {sev}")
            w(f": {first_sent}")
            ref_po = f.get("referencedPO", "")
            if ref_po:
                w(f": PO {ref_po}")
            w("")

    # ── Line Alignment ──
    portions = alignment.get("portions", [])
    if portions:
        w("## Line Alignment")
        w("")
        for portion in portions:
            po_id = portion.get("referencedPO", "?")
            bl = po_baselines.get(po_id, {})
            po_total = bl.get("total", 0)
            po_cur = bl.get("currency", "")

            matched = portion.get("matchedLines", [])
            unmatched = portion.get("unmatchedPO", [])
            phantom = portion.get("phantomASN", [])
            n_lines = len(matched)
            status_tag = "OK" if not unmatched and not phantom else "ISSUES"

            w(f"### PO {po_id} — {n_lines} line(s), {po_cur} {po_total:,.2f} [{status_tag}]")
            w("")

            if matched:
                w("| Line | Qty (PO / ASN) | Price (PO / ASN) | Status |")
                w("|------|---------------|-----------------|--------|")
                for m in matched:
                    po_data = m.get("po", {})
                    asn_data = m.get("asn", {})
                    po_line = m.get("poLine") or m.get("lineNumber", "")
                    po_qty = m.get("poQty") or po_data.get("quantity", "")
                    asn_qty = m.get("asnQty") or asn_data.get("quantity", "")
                    po_price = m.get("poPrice") or po_data.get("unitPrice", "")
                    asn_price = m.get("asnPrice") or asn_data.get("unitPrice", "")
                    m_cur = po_data.get("currency", po_cur)
                    if m.get("qtyMatch") is not None and m.get("priceMatch") is not None:
                        ok = m["qtyMatch"] and m["priceMatch"]
                    else:
                        ok = str(po_qty) == str(asn_qty) and str(po_price) == str(asn_price)
                    status = "OK" if ok else "MISMATCH"
                    w(f"| {po_line} | {po_qty} / {asn_qty} | {po_price} / {asn_price} {m_cur} | {status} |")
                w("")

            if unmatched:
                w("**Missing from ASN:**")
                for u in unmatched:
                    w(f"- Line {u.get('lineNumber', '?')}: {u.get('description', '')}")
                w("")

            if phantom:
                w("**Phantom lines (not in PO):**")
                for p in phantom:
                    w(f"- Line {p.get('lineNumber', '?')}: {p.get('description', '')}")
                w("")

    # ── Rules Summary (deduplicated: one row per rule ID, worst status wins) ──
    if rule_results:
        from collections import OrderedDict
        rule_map = OrderedDict()
        for r in sorted(rule_results, key=lambda x: x.get("rule_id", "")):
            rid = r.get("rule_id", "")
            if rid not in rule_map:
                rule_map[rid] = dict(r)
            elif r["status"] == "FAIL":
                rule_map[rid]["status"] = "FAIL"
                if r.get("detail"):
                    rule_map[rid]["detail"] = r["detail"]

        deduped = list(rule_map.values())
        pass_count = sum(1 for r in deduped if r["status"] == "PASS")
        fail_count = sum(1 for r in deduped if r["status"] == "FAIL")

        w("## Rules")
        w("")
        w(f"{pass_count} passed, {fail_count} failed out of {len(deduped)} rules.")
        w("")
        w("| Rule | Name | Result | Severity |")
        w("|------|------|--------|----------|")
        for r in deduped:
            status = r["status"]
            sev = r.get("severity", "")
            tag = "PASS" if status == "PASS" else "**FAIL**"
            w(f"| {r.get('rule_id', '')} | {r.get('name', '')} | {tag} | {sev} |")
        w("")

    # ── Footer ──
    w("---")
    w(f"*Pipeline: 7-stage LLM + Enforcer | {now}*")

    md_content = "\n".join(lines)
    out_path = OUTPUT / "reports" / f"ASN_{_safe_filename_id(shipment_id)}_report.md"
    out_path.write_text(md_content, encoding="utf-8")
    return out_path


# ══════════════════════════════════════════════════════════════
# PO File Lookup — finds PO files by orderID
# ══════════════════════════════════════════════════════════════

def find_po_files(po_ids: list) -> dict:
    """Search the DATA folder for PO files matching the given orderIDs.
    Returns dict: {orderID: filepath}

    Matches against both legacy `PO_*_XML.txt` files and `PO_*.xml`
    (synthesised cohort). Matching strategy, in order:
      1. Anchored: `PO_{id}_` somewhere in the filename — prevents
         substring collision (e.g. id "ORDER_12" must not match
         "PO_ORDER_1234_XML.txt").
      2. Synthesised-cohort fallback: if the orderID starts with
         `SYN_PO_`, strip that prefix and try again — covers cases
         where the ASN references `SYN_PO_SYN-M01` but the file is
         `PO_SYN-M01.xml`.

    Empty / blank IDs are skipped.
    """
    clean_ids = [pid for pid in (po_ids or []) if pid and pid.strip()]
    if not clean_ids:
        return {}

    found = {}
    candidates = list(DATA.rglob("PO_*_XML.txt")) + list(DATA.rglob("PO_*.xml"))
    for po_file in candidates:
        stem = po_file.stem  # e.g. PO_ORDER_SYN_0001 or PO_SYN-M01
        for po_id in clean_ids:
            if po_id in found:
                continue
            # Strategy 1: exact-anchored match
            if f"PO_{po_id}_" in po_file.name or stem.endswith(f"PO_{po_id}"):
                found[po_id] = po_file
                continue
            # Strategy 2: synthesised orderID has 'SYN_PO_' prefix that the
            # filename does not. Strip it and retry.
            if po_id.startswith("SYN_PO_"):
                short = po_id[len("SYN_PO_"):]
                if f"PO_{short}." in po_file.name or stem == f"PO_{short}":
                    found[po_id] = po_file
    return found


def _missing_po_result(
    asn_record: dict,
    missing_pos: list,
    found_pos: list | None = None,
) -> dict:
    """Build a fully-shaped pipeline result for the case where one or more
    referenced POs cannot be located in the local registry.

    The local PO file collection is only ~30 days deep, so older POs are
    expected to be missing. We do NOT crash; we return a REVIEW disposition
    with explicit action items so the caller (API / Tampermonkey / watcher)
    can render the result like any other pipeline output.
    """
    shipment_id = asn_record.get("shipmentID", "unknown")
    found_pos = found_pos or []
    referenced = list(asn_record.get("referencedPOs", []))

    all_missing = not found_pos
    reason = (
        "PO baseline unavailable -- local registry has no matching PO file(s). "
        "ASN cannot be validated against PO until the PO is fetched."
    ) if all_missing else (
        f"Partial PO baseline -- {len(missing_pos)} of {len(referenced)} "
        f"referenced PO(s) not found in local registry."
    )

    action_items = [{
        "source": "pipeline",
        "rule_id": "R10",
        "severity": "WARNING" if all_missing else "WARNING",
        "detail": (
            f"PO {pid} referenced by ASN {shipment_id} not found locally "
            f"(searched {DATA} for PO_{pid}_XML.txt). "
            f"Fetch the PO via Tampermonkey or Transaction Tracker and retry."
        ),
        "referencedPO": pid,
        "lineNumber": None,
    } for pid in missing_pos]

    summary = {
        "criticalIssues": 0,
        "warningIssues": len(missing_pos),
        "rulesTotal": 0,
        "rulesPass": 0,
        "rulesFail": 0,
        "semanticFindings": 0,
        "portions": len(referenced),
        "matchedLines": 0,
        "unmatchedPO": 0,
        "phantomASN": 0,
        "enforcerClean": True,
        "enforcerOverrides": 0,
        "missingPOs": missing_pos,
        "foundPOs": found_pos,
    }

    disposition = {
        "disposition": "REVIEW",
        "reason": reason,
        "summary": summary,
        "actionItems": action_items,
    }

    # Persist a minimal audit log so we still have a trail for these cases.
    try:
        audit_path = audit_log(
            asn_parsed={"shipmentID": shipment_id, "portions": []},
            po_baselines={pid: {} for pid in found_pos},
            validated_report={
                "ruleResults": [],
                "semanticFindings": [],
                "overallAssessment": reason,
                "enforcer_log": [],
            },
            disposition=disposition,
        )
    except Exception as e:
        print(f"  [WARNING] audit_log failed for missing-PO case: {e}")
        audit_path = None

    return {
        "shipmentID": shipment_id,
        "referencedPOs": referenced,
        "missingPOs": missing_pos,
        "foundPOs": found_pos,
        "disposition": "REVIEW",
        "reason": reason,
        "summary": summary,
        "actionItems": action_items,
        "overallAssessment": reason,
        "contextMetadata": {
            "shipmentID": shipment_id,
            "po_count": len(found_pos),
            "po_orderIDs": found_pos,
            "asn_portions": len(referenced),
            "asn_total_items": 0,
            "rules_injected": 0,
            "prompt_system_chars": 0,
            "prompt_user_chars": 0,
            "missing_po_count": len(missing_pos),
        },
        "auditLog": str(audit_path) if audit_path else "",
        "reportMarkdown": "",
    }


# ══════════════════════════════════════════════════════════════
# Full Pipeline Orchestrator
# ══════════════════════════════════════════════════════════════

def run_pipeline_from_xml(asn_xml: str, name="", enforcer_mode: str | None = None) -> dict:
    """Execute the full 7-stage pipeline from raw ASN XML string.
    Automatically finds referenced PO files from the DATA folder.

    enforcer_mode overrides the module-level ENFORCER_MODE for this call:
    "none" | "deterministic" (default) | "llm" | "tool_use" | "partition".
    """
    mode = enforcer_mode or ENFORCER_MODE
    print(f"\n{'='*60}")
    print(f"Pipeline: {name or 'API Request'}  [enforcer={mode}]")
    print(f"{'='*60}")

    # Stage 2: ASN Ingestion (from raw XML)
    print("\n[Stage 2] Ingesting ASN...")
    asn_record = ingest_asn_from_xml(asn_xml)
    print(f"  Shipment {asn_record['shipmentID']} -> PO(s): {asn_record['referencedPOs']}")

    # Stage 1: Find and parse referenced PO files
    print("\n[Stage 1] Parsing PO baselines...")
    referenced = asn_record["referencedPOs"]
    if not referenced:
        print("  [WARNING] ASN has no referencedPOs -- nothing to validate against.")
        return _missing_po_result(asn_record, missing_pos=[], found_pos=[])

    po_files = find_po_files(referenced)
    missing_pos = [po for po in referenced if po not in po_files]
    if missing_pos:
        print(f"  [WARNING] PO files not found for: {missing_pos}")

    po_baselines = {}
    for po_id, po_path in po_files.items():
        try:
            bl = parse_po_cxml(po_path)
        except Exception as e:
            print(f"  [ERROR] Failed to parse PO {po_id} ({po_path.name}): "
                  f"{type(e).__name__}: {e}")
            missing_pos.append(po_id)
            continue
        po_baselines[bl["orderID"]] = bl
        (OUTPUT / "baselines" / f"{bl['orderID']}.json").write_text(
            json.dumps(bl, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  PO {bl['orderID']}: {len(bl['lineItems'])} lines, {bl['currency']} {bl['total']}")

    if not po_baselines:
        # Either no files matched or every file failed to parse -- same result.
        return _missing_po_result(
            asn_record,
            missing_pos=missing_pos or list(referenced),
            found_pos=[],
        )

    # Stage 3: LLM Parse ASN
    print("\n[Stage 3] LLM parsing ASN...")
    asn_parsed = llm_parse_asn(asn_record["raw_xml"])
    # The LLM parser does not extract supplier credentials — copy the
    # deterministic Stage 2 ingest's supplierVendorID/NetworkID across so
    # downstream code (R09 validation, _load_supplier_profile RAG lookup)
    # has them.
    asn_parsed.setdefault("supplierVendorID", asn_record.get("supplierVendorID", ""))
    asn_parsed.setdefault("supplierNetworkID", asn_record.get("supplierNetworkID", ""))
    shipment_id = asn_parsed.get("shipmentID", "unknown")
    (OUTPUT / "parsed" / f"ASN_{_safe_filename_id(shipment_id)}_parsed.json").write_text(
        json.dumps(asn_parsed, indent=2, ensure_ascii=False), encoding="utf-8")
    total_items = sum(len(p.get("items", [])) for p in asn_parsed.get("portions", []))
    print(f"  Parsed {len(asn_parsed.get('portions', []))} portion(s), {total_items} item(s)")

    master_prompt = {"context_metadata": {}}
    if mode == "partition":
        # Mode E: Stage 5a (deterministic owned-rule verdicts) + Stage 5b
        # (LLM-narrow over the 14 non-owned rules) + trivial merge.  Bypasses
        # the standard build_validation_prompt + llm_validate + _dispatch_enforcer
        # path because the merge is performed inside partition_validate_and_enforce.
        print("\n[Stage 4/5/6] Partition mode (deterministic + LLM-narrow + merge)...")
        validated_report = partition_validate_and_enforce(po_baselines, asn_parsed)
        # Surface the narrow-prompt context_metadata (RAG audit fields:
        # supplier_profile_used, etc.) into the result, mirroring the
        # non-partition path so the RAG ablation table can count injections.
        master_prompt = {"context_metadata": validated_report.get("_context_metadata", {})}
        (OUTPUT / "reports" / f"ASN_{_safe_filename_id(shipment_id)}_master_report.json").write_text(
            json.dumps(validated_report, indent=2, ensure_ascii=False), encoding="utf-8")
        rule_results = validated_report.get("ruleResults", [])
        print(f"  Rules: {sum(1 for r in rule_results if r['status']=='PASS')} pass,"
              f" {sum(1 for r in rule_results if r['status']=='FAIL')} fail "
              f"({len(rule_results)} entries)")
    else:
        # Stage 4: Prompt Builder (mode-aware: tool_use prepends tool-use instructions)
        print("\n[Stage 4] Building master validation prompt...")
        master_prompt = build_validation_prompt(po_baselines, asn_parsed,
                                                 use_tools=(mode == "tool_use"))
        total_chars = master_prompt["context_metadata"]["prompt_system_chars"] + master_prompt["context_metadata"]["prompt_user_chars"]
        print(f"  Prompt: {total_chars:,} chars ({master_prompt['context_metadata']['rules_injected']} rules)")

        # Stage 5: LLM Validation (tool_use flips on in-generation function calling)
        use_tools = (mode == "tool_use")
        print(f"\n[Stage 5] LLM validation (alignment + rules + semantic)"
              + (" [tools enabled]" if use_tools else "") + "...")
        master_report = llm_validate(master_prompt, use_tools=use_tools)
        (OUTPUT / "reports" / f"ASN_{_safe_filename_id(shipment_id)}_master_report.json").write_text(
            json.dumps(master_report, indent=2, ensure_ascii=False), encoding="utf-8")
        rule_results = master_report.get("ruleResults", [])
        print(f"  Rules: {sum(1 for r in rule_results if r['status']=='PASS')} pass,"
              f" {sum(1 for r in rule_results if r['status']=='FAIL')} fail")
        if use_tools:
            print(f"  Tool calls: {len(master_report.get('tool_calls_log', []))}")

        # Stage 6: Output Enforcer (dispatched by mode)
        print(f"\n[Stage 6] Enforcing output (mode={mode})...")
        validated_report = _dispatch_enforcer(mode, master_report, po_baselines, asn_parsed)
    (OUTPUT / "reports" / f"ASN_{_safe_filename_id(shipment_id)}_validated.json").write_text(
        json.dumps(validated_report, indent=2, ensure_ascii=False), encoding="utf-8")
    enforcer = validated_report.get("enforcer_summary", {})
    print(f"  Enforcer clean: {enforcer.get('clean', True)} (overrides: {enforcer.get('overrides', 0)})")

    # Stage 7: Decision Router + Audit
    print("\n[Stage 7] Computing disposition...")
    disposition = compute_disposition(validated_report)

    # Surface missing POs (partial-baseline case): inject action items + bump
    # disposition to at least REVIEW so the caller sees the gap.
    if missing_pos:
        for pid in missing_pos:
            disposition["actionItems"].append({
                "source": "pipeline",
                "rule_id": "R10",
                "severity": "WARNING",
                "detail": (
                    f"PO {pid} referenced by ASN {shipment_id} not found in "
                    f"local registry -- validated against {len(po_baselines)} "
                    f"of {len(referenced)} PO(s) only."
                ),
                "referencedPO": pid,
                "lineNumber": None,
            })
        disposition["summary"]["missingPOs"] = missing_pos
        disposition["summary"]["warningIssues"] += len(missing_pos)
        if disposition["disposition"] == "PASS":
            disposition["disposition"] = "REVIEW"
            disposition["reason"] = (
                f"Partial PO baseline -- {len(missing_pos)} PO(s) missing from "
                f"local registry; validated portions passed."
            )

    audit_path = audit_log(asn_parsed, po_baselines, validated_report, disposition)
    print(f"  >>> DISPOSITION: {disposition['disposition']} -- {disposition['reason']}")

    # Stage 7c: Generate Markdown Report (human-readable)
    md_path = generate_markdown_report(asn_parsed, po_baselines, validated_report, disposition)
    print(f"  Report: {md_path}")

    return {
        "shipmentID": shipment_id,
        "referencedPOs": list(po_baselines.keys()),
        "missingPOs": missing_pos,
        "disposition": disposition["disposition"],
        "reason": disposition["reason"],
        "summary": disposition["summary"],
        "actionItems": disposition["actionItems"],
        "overallAssessment": validated_report.get("overallAssessment", ""),
        "contextMetadata": master_prompt["context_metadata"],
        "auditLog": str(audit_path),
        "reportMarkdown": str(md_path),
    }
