"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║         C S L F   E N G I N E   v 5 . 5                                         ║
║         Cognitive Stability & Legitimacy Framework                               ║
║                                                                                  ║
║         Author   : Dr. Elhabib Kherroubi                                        ║
║         Version  : 5.5 — Sovereign Final | R-AGAM Interface-Compliant           ║
║         Protocol : SRIP v1.0 | Annex 1 Schema | SBS §10 | SHS §11              ║
║         Date     : April 2026                                                    ║
║                                                                                  ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║                                                                                  ║
║  SOVEREIGN ARCHITECTURE:                                                         ║
║                                                                                  ║
║   Reasoning Path (natural language)                                              ║
║         │                                                                        ║
║         ▼                                                                        ║
║   ┌─────────────────────────────────────────────┐                               ║
║   │  CSLF EPISTEMIC CORE                        │                               ║
║   │                                             │                               ║
║   │  [1] NLP Classifier → {stable,drift,chaos}  │                               ║
║   │  [2] Lyapunov Stability Engine              │                               ║
║   │       K(t) = 1 − H(p)/log(n)               │                               ║
║   │       V(t) = −log(K+ε)                      │                               ║
║   │       H(t) = dI·exp(λ·dV)  [hazard]         │                               ║
║   │  [3] KLL Engine  (Whitepaper §4.2 exact)    │                               ║
║   │       KLL = (SA × CI × SV × EG)^w           │                               ║
║   │  [4] FAC — Final Admissibility Condition     │                               ║
║   │       K_avg≥θ_K  ∧  H_max≤H_CRIT  ∧  KLL>0 │                               ║
║   │  [5] S_signal — Sovereign Stability Signal  │                               ║
║   └─────────────────────────────────────────────┘                               ║
║         │                                                                        ║
║         ▼  (sealed — internal state never exposed)                               ║
║   ┌─────────────────────────────────────────────┐                               ║
║   │  CAV — Cognitive Admissibility Vector        │  ← Annex 1 exact schema     ║
║   │  {header, epistemic_payload, crypto_seal}   │                               ║
║   │   status: PASSED | FAILED_* | NULL_STATE     │                               ║
║   │   vectors: {K, H, KLL, S_signal}            │                               ║
║   │   flags: [epistemic anomaly list]            │                               ║
║   │   seal: SHA-256 + HMAC-SHA256 + pubkey_ref  │                               ║
║   └─────────────────────────────────────────────┘                               ║
║         │                                                                        ║
║         ▼  (opaque — R-AGAM sees ONLY this)                                      ║
║   R-AGAM Hard Gate (Sovereign Authorization Layer)                               ║
║   → Commit | Deny                                                                ║
║                                                                                  ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║                                                                                  ║
║  KEY COMPLIANCE FIXES vs v5.3/v5.4:                                             ║
║  ✓ KLL = (SA×CI×SV×EG)^w  — Whitepaper §4.2, SV restored                       ║
║  ✓ finalize() over full path  — SBS §10.3, SHS §11                              ║
║  ✓ THETA_K = 0.30  — calibrated to K_stable≈0.43, K_drift≈0.13                 ║
║  ✓ CAV exposes {K,H,KLL,S_signal} only — Annex 1 schema                         ║
║  ✓ Internal decomp (SA,CI,SV,EG,trace) NEVER transmitted — SBS §10.3            ║
║  ✓ NULL_STATE as epistemic non-existence, not error                              ║
║  ✓ HMAC-SHA256 sovereign seal (stdlib, no external deps)                         ║
║  ✓ S_signal = composite sovereign stability index                                ║
║  ✓ Transition entropy for epistemic drift detection                              ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import math, json, hashlib, hmac, secrets, uuid, time, re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════════
# §1  SOVEREIGN CONFIGURATION  — immutable, whitepaper-locked
# ════════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SovereignConfig:
    """
    Immutable configuration. All values per Whitepaper §4 and Technical Annexes.
    Any runtime modification constitutes an architectural violation.

    Calibration rationale:
      K_stable ≈ 0.434  (dist=[0.78,0.14,0.05,0.03])
      K_drift  ≈ 0.075  (dist=[0.50,0.25,0.15,0.10])
      K_chaos  ≈ 0.000  (dist=[0.25,0.25,0.25,0.25])
      THETA_K  = 0.30   → accepts stable-dominant paths, rejects pure-drift
    """
    LAMBDA:   float = 5.5    # Lyapunov amplification factor §4.1
    H_CRIT:   float = 4.5   # Hazard ceiling §4.1
    KLL_MIN:  float = 0.40   # Legitimacy threshold §4.2
    THETA_K:  float = 0.30   # Coherence threshold §4.5 FAC
    WINDOW:   int   = 5      # Memory window for SA/CI
    DECAY:    float = 0.85   # Exponential decay weight
    KLL_W:    float = 0.25   # KLL exponent (w in whitepaper)
    EPS:      float = 1e-9   # Numerical stability


CFG = SovereignConfig()


# ════════════════════════════════════════════════════════════════════════════════
# §2  SESSION SIGNING KEY  — generated once per session (sovereign seal)
# ════════════════════════════════════════════════════════════════════════════════

# In production: loaded from HSM. Here: HMAC-SHA256 (stdlib, cryptographically sound)
_SIGNING_SECRET: bytes = secrets.token_bytes(32)
_PUBLIC_KEY_REF: str   = "CSLF_PUB_" + hashlib.sha256(_SIGNING_SECRET).hexdigest()[:16].upper()


def _sovereign_sign(payload_bytes: bytes) -> str:
    """HMAC-SHA256 sovereign signature — non-forgeable without _SIGNING_SECRET."""
    return hmac.new(_SIGNING_SECRET, payload_bytes, hashlib.sha256).hexdigest()


def _sovereign_verify(payload_bytes: bytes, signature: str) -> bool:
    """Constant-time verification — R-AGAM Hard Gate check."""
    expected = _sovereign_sign(payload_bytes)
    return hmac.compare_digest(expected, signature)


# ════════════════════════════════════════════════════════════════════════════════
# §3  NLP CLASSIFIER  — deterministic, domain-calibrated
# ════════════════════════════════════════════════════════════════════════════════

LEXICON: Dict[str, List[str]] = {
    "chaos": [
        "random", "unknown", "unreliable", "hallucination", "undefined",
        "corrupt", "arbitrary", "nonsense", "invalid", "garbage",
        "meaningless", "incoherent", "broken", "noise"
    ],
    "evidence": [
        "data", "evidence", "because", "therefore", "shows", "confirms",
        "proves", "demonstrates", "analysis", "results", "indicates",
        "measured", "verified", "established", "supported", "grounded",
        "documented", "recorded", "according", "study", "research",
        "found", "conclude", "observed", "reported", "tested"
    ],
    "uncertainty": [
        "maybe", "perhaps", "likely", "could", "might", "possibly",
        "uncertain", "unclear", "seems", "appears", "probably",
        "supposedly", "allegedly", "arguably", "presumably"
    ],
    "subject_markers": [
        "the", "this", "that", "it", "we", "i", "they", "our",
        "data", "evidence", "analysis", "system", "model", "results",
        "study", "report", "research", "findings", "outcome"
    ],
    "negation": [
        "not", "never", "no", "neither", "nor", "cannot",
        "won't", "doesn't", "isn't", "aren't", "wasn't", "weren't"
    ]
}


def classify(text: str) -> str:
    """
    Deterministic state classifier.
    Priority: chaos > evidence > uncertainty > drift (default).
    Single chaos word collapses entire step — per Whitepaper §4.1 hazard model.
    """
    t = text.lower()
    if any(w in t for w in LEXICON["chaos"]):    return "chaos"
    if any(w in t for w in LEXICON["evidence"]): return "stable"
    if any(w in t for w in LEXICON["uncertainty"]): return "drift"
    return "drift"


def parse_steps(text: str) -> List[str]:
    """Split reasoning text into discrete evaluation steps."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        lines = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    return lines


# ════════════════════════════════════════════════════════════════════════════════
# §4  PROBABILITY DISTRIBUTIONS  — fixed, deterministic K values
# ════════════════════════════════════════════════════════════════════════════════

DISTRIBUTIONS: Dict[str, List[float]] = {
    "stable": [0.78, 0.14, 0.05, 0.03],   # K ≈ 0.434
    "drift":  [0.50, 0.25, 0.15, 0.10],   # K ≈ 0.075
    "chaos":  [0.25, 0.25, 0.25, 0.25],   # K ≈ 0.000 (max entropy)
}


def _entropy(p: List[float]) -> float:
    return -sum(x * math.log(x + CFG.EPS) for x in p)


# ════════════════════════════════════════════════════════════════════════════════
# §5  SEMANTIC VALIDITY (SV)  — restored per Whitepaper §4.2
# ════════════════════════════════════════════════════════════════════════════════

def compute_sv(text: str, state: str) -> float:
    """
    SV = Semantic Validity — structural coherence of the reasoning unit.

    Whitepaper §4.2: SV measures whether the epistemic unit is
    semantically well-formed, not merely syntactically present.

    Criteria applied:
      - Word count (structural minimum)
      - Subject presence (referential grounding)
      - Semantic conflict (chaos + evidence = self-contradiction → SV collapse)
      - Negation of evidence (partial penalty — legitimate construction)
    """
    words = text.lower().split()
    n = len(words)

    if n < 3:
        return 0.15  # structurally degenerate

    has_subject  = any(w in words for w in LEXICON["subject_markers"])
    has_evidence = any(w in text.lower() for w in LEXICON["evidence"])
    has_chaos    = any(w in text.lower() for w in LEXICON["chaos"])
    has_negation = any(w in words for w in LEXICON["negation"])

    base = 0.88 if has_subject else 0.52

    if has_chaos and has_evidence:
        base *= 0.35   # semantic contradiction: collapses SV
    if has_negation and has_evidence:
        base *= 0.78   # negation of evidence: mild penalty
    if n <= 4:
        base *= 0.72   # very short: limited semantic content

    return round(min(1.0, max(0.0, base)), 4)


# ════════════════════════════════════════════════════════════════════════════════
# §6  CORE ENGINE  — CSLFv55
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class _StepRecord:
    """Internal step trace — NEVER transmitted to R-AGAM (SBS §10.3)."""
    index:  int
    text:   str
    state:  str
    K:      float
    H:      float
    SA:     float
    CI:     float
    SV:     float
    EG:     float
    KLL:    float


class CSLFv55:
    """
    CSLF Engine v5.5 — Sovereign Epistemic Core.

    Contract:
      - step()     : evaluate one reasoning unit
      - finalize() : assess full path, produce internal result
      - seal_cav() : produce R-AGAM-ready CAV artifact (sealed, opaque)
      - verify()   : R-AGAM Hard Gate verification method

    The engine is single-use: once finalize() is called, it is sealed.
    No re-evaluation, no backflow, no state mutation after sealing.
    """

    def __init__(self):
        self._V_prev:     float       = 0.0
        self._I_prev:     float       = 0.0
        self._K_hist:     List[float] = []
        self._dV_hist:    List[float] = []
        self._state_seq:  List[str]   = []
        self._trace:      List[_StepRecord] = []
        self._sealed:     bool        = False
        self._result:     Optional[Dict] = None

    # ── per-step evaluation ───────────────────────────────────────────────────

    def step(self, index: int, text: str) -> _StepRecord:
        if self._sealed:
            raise RuntimeError("Engine sealed — epistemic closure enforced.")

        state = classify(text)
        p     = DISTRIBUTIONS[state]

        # ── Coherence K(t)  [Lyapunov energy] ────────────────────────────────
        Hs = _entropy(p)
        K  = max(CFG.EPS, 1.0 - Hs / math.log(len(p)))
        V  = -math.log(K + CFG.EPS)
        dV = max(0.0, V - self._V_prev)
        self._V_prev = V

        # ── Intent formation I(t) ─────────────────────────────────────────────
        I  = K + (0.35 if state == "chaos" else 0.20 if state == "drift" else 0.0)
        I  = min(1.0, I)
        dI = max(0.0, I - self._I_prev)
        self._I_prev = I

        # ── Hazard H(t)  [normalized, capped at 3×H_CRIT] ────────────────────
        raw_H = (dI + 0.05) * math.exp(CFG.LAMBDA * (dV + 0.02))
        H     = min(CFG.H_CRIT * 3.0, math.log1p(raw_H))

        # ── Memory  [exponentially weighted window] ───────────────────────────
        self._K_hist.append(K)
        self._dV_hist.append(dV)
        self._state_seq.append(state)
        if len(self._K_hist) > CFG.WINDOW:
            self._K_hist.pop(0)
            self._dV_hist.pop(0)

        weights = [CFG.DECAY ** i for i in range(len(self._K_hist))][::-1]
        W = sum(weights)

        SA = sum(k * w for k, w in zip(self._K_hist, weights)) / W
        CI = max(0.0, 1.0 - sum(d * w for d, w in zip(self._dV_hist, weights)) / W)

        # ── SV — Semantic Validity  [Whitepaper §4.2 — restored] ─────────────
        SV = compute_sv(text, state)

        # ── EG — Evidence Grounding ───────────────────────────────────────────
        if any(w in text.lower() for w in LEXICON["evidence"]):
            EG = 1.00
        elif any(w in text.lower() for w in LEXICON["chaos"]):
            EG = 0.20
        elif any(w in text.lower() for w in LEXICON["uncertainty"]):
            EG = 0.52
        else:
            EG = 0.65

        # ── KLL  [Whitepaper §4.2 exact: (SA×CI×SV×EG)^w] ───────────────────
        KLL = (SA * CI * SV * EG) ** CFG.KLL_W

        record = _StepRecord(
            index=index, text=text, state=state,
            K=round(K,4), H=round(H,4),
            SA=round(SA,4), CI=round(CI,4),
            SV=round(SV,4), EG=round(EG,4),
            KLL=round(KLL,4)
        )
        self._trace.append(record)
        return record

    # ── path finalization ─────────────────────────────────────────────────────

    def finalize(self) -> Dict:
        """
        FAC — Final Admissibility Condition (Whitepaper §4.5).
        Evaluated over FULL PATH — SBS §10.3, SHS §11.

        Strict ordering (per Annex 1 §4 terminal states):
          1. H_max > H_CRIT           → FAILED_HAZARD
          2. K_avg < THETA_K          → FAILED_COHERENCE
          3. KLL_final < KLL_MIN      → FAILED_LEGITIMACY
          4. K≈0 ∧ KLL≈0             → NULL_STATE (epistemic non-existence)
          5. All pass                 → PASSED
        """
        assert len(self._trace) > 0, "No steps evaluated."
        self._sealed = True

        # Path-level metrics
        weights = [CFG.DECAY ** i for i in range(len(self._trace))][::-1]
        W       = sum(weights)
        K_avg   = sum(s.K * w for s, w in zip(self._trace, weights)) / W
        H_max   = max(s.H for s in self._trace)
        KLL_fin = self._trace[-1].KLL

        # Transition entropy — epistemic drift index
        transitions: Dict[str, int] = {}
        for i in range(1, len(self._state_seq)):
            key = f"{self._state_seq[i-1]}→{self._state_seq[i]}"
            transitions[key] = transitions.get(key, 0) + 1
        total_tr = sum(transitions.values()) or 1
        T_ent = -sum((v/total_tr) * math.log(v/total_tr + CFG.EPS)
                      for v in transitions.values())

        # S_signal — Sovereign Stability Signal (Annex 1 schema field)
        # Composite: coherence × (1 - normalized hazard) × legitimacy
        H_norm   = min(1.0, H_max / (CFG.H_CRIT * 3))
        S_signal = round(K_avg * (1.0 - H_norm) * KLL_fin, 4)

        # FAC evaluation
        if H_max > CFG.H_CRIT:
            verdict = "FAILED_HAZARD"
        elif K_avg < CFG.THETA_K:
            verdict = "FAILED_COHERENCE"
        elif KLL_fin < CFG.KLL_MIN:
            verdict = "FAILED_LEGITIMACY"
        elif K_avg < CFG.EPS and KLL_fin < CFG.EPS:
            verdict = "NULL_STATE"
        else:
            verdict = "PASSED"

        self._result = {
            "verdict":      verdict,
            "K_avg":        round(K_avg, 4),
            "H_max":        round(H_max, 4),
            "KLL_final":    round(KLL_fin, 4),
            "S_signal":     S_signal,
            "T_entropy":    round(T_ent, 4),
            "steps":        len(self._trace),
            "state_seq":    self._state_seq[:],
            "transitions":  transitions,
        }
        return self._result

    # ── CAV production  [Annex 1 exact schema] ───────────────────────────────

    def seal_cav(self) -> Dict:
        """
        Produce the sealed CAV artifact per SRIP v1.0 + Annex 1.

        ┌─────────────────────────────────────────────────────────┐
        │  CAV STRUCTURE (Annex 1 §2 — exact)                    │
        │  {                                                      │
        │    "header": {artifact_id, timestamp_utc, origin}      │
        │    "epistemic_payload": {                               │
        │      "status": PASSED|FAILED_*|NULL_STATE              │
        │      "vectors": {K, H, KLL, S_signal}   ← Annex 1     │
        │      "flags": [anomaly list]                           │
        │    }                                                    │
        │    "cryptographic_seal": {hash, signature, pub_key}    │
        │  }                                                      │
        │                                                         │
        │  WHAT R-AGAM SEES:  status + vectors + flags + seal    │
        │  WHAT R-AGAM NEVER SEES: SA, CI, SV, EG, trace, steps  │
        └─────────────────────────────────────────────────────────┘
        """
        assert self._sealed and self._result, "Must call finalize() first."

        r = self._result
        artifact_id = f"CAV-2026-{uuid.uuid4().hex[:8].upper()}"
        timestamp   = datetime.now(timezone.utc).isoformat()

        # Epistemic anomaly flags
        flags: List[str] = []
        if r["H_max"] > CFG.H_CRIT:
