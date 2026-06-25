# Validation rule register (R01–R22)

The pipeline validates an Advanced Shipping Notice (ASN) against its originating
Purchase Order (PO) with 22 rules. Each rule returns a verdict (`ok` / `fail`)
and a severity (`info` / `warn` / `crit`). The document disposition is the
reduction of all rule outputs: one critical failure forces **REJECT**, one
warning failure forces **REVIEW**, otherwise **PASS**.

Each rule is assigned to one of two regimes:

- **Deterministic** — the check reduces to a closed-form computation on fields
  the pipeline already holds, so the enforcer recomputes it in code and holds
  authority over the verdict. Eight rules: R01, R02, R03, R06, R15, R16, R18, R19.
- **Language model** — the check needs interpretation, so the model owns the
  verdict. The remaining fourteen rules.

| Rule | Name | Category | Severity | Check (one line) | Regime |
|---|---|---|---|---|---|
| R01 | Shipped quantity match | Arithmetic | Critical | ASN quantity within the PO tolerance band | Deterministic |
| R02 | Unit price match | Arithmetic | Critical | ASN unit price within the PO tolerance band | Deterministic |
| R03 | Delivery date feasibility | Arithmetic | Warning | ASN delivery date within the PO tolerance window | Deterministic |
| R04 | Shipment date realism | Arithmetic | Warning | noticeDate ≤ shipmentDate ≤ deliveryDate, logically coherent | Language model |
| R05 | UOM consistency | Structural | Critical | ASN unit of measure matches the PO (canonical equivalence) | Language model |
| R06 | Line item completeness | Structural | Critical (downgr.) | Every PO line in the portion has a matching ASN item | Deterministic |
| R07 | No phantom lines | Structural | Critical | ASN has no item referencing a non-existent PO line | Language model |
| R08 | Currency match | Structural | Critical | ASN currency matches the PO at header and line level | Language model |
| R09 | Supplier ID validation | Structural | Critical | ASN supplier network ID matches the PO supplier network ID | Language model |
| R10 | PO reference integrity | Structural | Critical | Each portion references a valid, existing PO orderID | Language model |
| R11 | Schema compliance | Structural | Informational | ASN is valid cXML and parses without errors | Language model |
| R12 | Mandatory field presence | Structural | Warning | Required ASN header and item fields are present | Language model |
| R13 | Transport terms match | Semantic | Warning | ASN incoterms match the PO incoterms | Language model |
| R14 | Ship-to address match | Semantic | Warning | ASN ship-to address matches the PO ship-to address | Language model |
| R15 | Total value reconciliation | Arithmetic | Warning | quantity × unitPrice matches the PO line total within tolerance | Deterministic |
| R16 | Partial-shipment handling | Arithmetic | Info (escalates) | Partial shipment flagged; escalates when R01 detects under-ship | Deterministic |
| R17 | Multi-PO consolidation | Structural | Warning | Cross-PO boundary integrity for multi-PO ASNs | Language model |
| R18 | Duplicate ASN detection | Structural | Critical | shipmentID + PO set not previously seen (persisted state) | Deterministic |
| R19 | ShipmentID length | Structural | Critical | shipmentID ≤ 35 characters | Deterministic |
| R20 | Weight plausibility | Arithmetic | Warning | grossWeight > netWeight and grossWeight ≤ netWeight × 10 | Language model |
| R21 | Mandatory packaging fields | Structural | Critical | Required packaging fields are present | Language model |
| R22 | Order fulfilment status | Arithmetic | Warning | Total notified quantity does not exceed the PO quantity | Language model |

## The five designs

The deterministic enforcer can sit at three positions, and two further designs
sit outside the pattern. The pipeline mode keys map to the dissertation as:

| key | design | enforcer | what it does |
|---|---|---|---|
| `none` | pure LLM (Mode A) | no | the model evaluates every rule; nothing post-processes it |
| `deterministic` | post hoc (Mode B) | yes | the model evaluates everything, the enforcer overrides the 8 owned rules at Stage 6 |
| `llm` | LLM-as-judge (Mode C) | no | a second model call audits the first at Stage 6 instead of a deterministic check |
| `tool_use` | in generation (Mode D) | yes | the deterministic checks are exposed as tools the model calls during Stage 5 |
| `partition` | pre hoc (Mode E) | yes | the deterministic rules are settled first; the model is asked only about the other 14 |

R18 needs cross-document state (the set of previously seen notices), so it is
always settled by the deterministic side, never by the model.
