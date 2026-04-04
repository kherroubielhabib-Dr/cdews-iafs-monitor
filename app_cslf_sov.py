"""
================================================================================
CSLF SOVEREIGN SIGNAL LAYER
Cognitive Stability & Legitimacy Framework
Author: Dr. Elhabib Kherroubi
Version: 1.0 (MVP - Ready for R-AGAM Handoff)
================================================================================
Core Principle: "A decision is not judged by its correctness,
                but by the stability and legitimacy of its formation."
Output: CAV Artifact (Cryptographically Sealed Epistemic Object)
================================================================================
"""

import streamlit as st
import math
import pandas as pd
import json
import hashlib
import time
import random
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class CSLFConfig:
    LAMBDA: float = 8.0
    H_CRIT: float = 0.6
    KLL_MIN: float = 0.4
    KLL_WINDOW: int = 5
    DOMAIN_RISK: float = 0.7
    W_SA: float = 0.30
    W_CI: float = 0.30
    W_SV: float = 0.25
    W_EG: float = 0.15
    ALPHA: float = 0.4
    BETA: float = 0.3
    GAMMA: float = 0.3
    P_EXP: float = 2.0

CONFIG = CSLFConfig()

EVIDENCE_KEYWORDS = [
    "data", "evidence", "because", "therefore", "income", "debt", "score",
    "stable", "fact", "research", "study", "according", "shows", "indicates"
]

# ==============================================================================
# UTILITIES
# ==============================================================================

def entropy(probs: List[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)

def normalize_entropy(h: float, n: int) -> float:
    return h / math.log(n) if n > 1 else 0.0

def safe_pow(base: float, exp: float) -> float:
    return math.pow(max(1e-9, base), exp)

def generate_probs(state: str, step: int = 0) -> List[float]:
    s = state.lower()
    if s == "stable":
        return [0.85, 0.10, 0.03, 0.02]
    elif s == "drift":
        return [0.55, 0.25, 0.12, 0.08]
    elif s == "chaos":
        base = min(0.33, 0.20 + 0.015 * step)
        return [base, base, base, max(0.01, 1 - 3 * base)]
    return [0.40, 0.30, 0.20, 0.10]

def get_token(state: str, step: int) -> str:
    tokens = {
        "stable": ["data", "evidence", "because", "fact", "income stable"],
        "drift": ["could", "perhaps", "maybe", "unclear"],
        "chaos": ["unknown", "random", "unreliable", "invented"]
    }
    lst = tokens.get(state.lower(), ["?"])
    return lst[step % len(lst)]

def sign_cav(cav: dict) -> str:
    raw = json.dumps(cav, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

# ==============================================================================
# CORE ENGINES
# ==============================================================================

class StabilityEngine:
    def __init__(self):
        self.V_prev = 0.0

    def update(self, probs: List[float]) -> Tuple[float, float, float]:
        h = entropy(probs)
        K = max(1e-9, 1 - normalize_entropy(h, len(probs)))
        V = -math.log(K + 1e-9)
        dV = max(0.0, V - self.V_prev)
        self.V_prev = V
        return K, V, dV

class IntentEngine:
    def __init__(self):
        self.I_prev = 0.0

    def update(self, probs: List[float], token: str) -> Tuple[float, float]:
        h = entropy(probs)
        I = 1 - normalize_entropy(h, len(probs))
        if token == "chaos":
            I = min(1.0, self.I_prev + 0.5)
        dI = max(0.0, I - self.I_prev)
        self.I_prev = I
        return I, dI

class KLLEngine:
    def __init__(self, config: CSLFConfig):
        self.config = config
        self.K_hist: List[float] = []
        self.dV_hist: List[float] = []

    def update(self, token: str, K: float, dV: float) -> Tuple[float, float, float, float, float]:
        self.K_hist.append(K)
        self.dV_hist.append(dV)
        if len(self.K_hist) > self.config.KLL_WINDOW:
            self.K_hist.pop(0)
            self.dV_hist.pop(0)
        SA = sum(self.K_hist) / len(self.K_hist)
        CI = max(0.0, 1 - sum(self.dV_hist) / len(self.dV_hist))
        SV = K
        EG = 1.0 if any(kw in token.lower() for kw in EVIDENCE_KEYWORDS) else 0.4
        score = (safe_pow(SA, self.config.W_SA) *
                 safe_pow(CI, self.config.W_CI) *
                 safe_pow(SV, self.config.W_SV) *
                 safe_pow(EG, self.config.W_EG))
        return score, SA, CI, SV, EG

class CSLFSignalEngine:
    def __init__(self, config: CSLFConfig = None):
        self.config = config or CONFIG
        self.stability = StabilityEngine()
        self.intent = IntentEngine()
        self.kll = KLLEngine(self.config)

    def evaluate_step(self, probs: List[float], token: str) -> Dict:
        K, V, dV = self.stability.update(probs)
        I, dI = self.intent.update(probs, token)
        H = dI * math.exp(self.config.LAMBDA * dV)
        CTL = (1 - K) + dV
        KLL_score, SA, CI, SV, EG = self.kll.update(token, K, dV)
        CTL_star = CTL * (1 + self.config.LAMBDA * (1 - KLL_score))
        inner = (self.config.ALPHA * H +
                 self.config.BETA * dV +
                 self.config.GAMMA * (1 - K) +
                 self.config.DOMAIN_RISK * safe_pow(CTL_star, self.config.P_EXP))
        S_signal = math.exp(-inner)
        return {
            "K": K, "dV": dV, "H": H, "KLL": KLL_score,
            "CTL_star": CTL_star, "S_signal": S_signal,
            "SA": SA, "CI": CI, "SV": SV, "EG": EG
        }

    def run(self, steps: List[Tuple[str, str]]) -> Dict:
        trace = []
        for idx, (state, token) in enumerate(steps):
            probs = generate_probs(state, idx)
            r = self.evaluate_step(probs, token)
            trace.append({
                "step": idx + 1, "state": state, "token": token,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
            })
        final = trace[-1]
        K_f, H_f, KLL_f = final["K"], final["H"], final["KLL"]
        flags = []
        if H_f > 1000:
            flags.append("hazard_explosion")
        if KLL_f < CONFIG.KLL_MIN:
            flags.append("low_legitimacy")
        if K_f < 0.01:
            flags.append("coherence_collapse")
        if "hazard_explosion" in flags:
            status = "FAILED_HAZARD"
        elif "low_legitimacy" in flags:
            status = "FAILED_LEGITIMACY"
        elif "coherence_collapse" in flags:
            status = "FAILED_COHERENCE"
        else:
            status = "PASSED"
        cav = {
            "header": {
                "artifact_type": "CAV_EPISTEMIC_CLOSURE",
                "version": "1.0",
                "cav_id": f"cav_{int(time.time())}_{random.randint(1000,9999)}",
                "issuer_id": "CSLF_ENGINE_DR_KHERROUBI",
                "target_layer": "R_AGAM_SOVEREIGN_GATE",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "domain": "financial"
            },
            "epistemic_closure": {
                "cav_status": status,
                "epistemic_meaning": {
                    "PASSED": "Epistemically admissible. Ready for sovereign resolution.",
                    "FAILED_HAZARD": "Instability explosion detected. Reasoning collapsed.",
                    "FAILED_LEGITIMACY": "Epistemic illegitimacy. Decision lacks right to exist.",
                    "FAILED_COHERENCE": "Coherence collapse. Decision disintegrated.",
                    "NULL_STATE": "Epistemic null state. No decision exists."
                }.get(status, "Unknown"),
                "execution_semantics": "ALLOW_IF_VERIFIED" if status == "PASSED" else "BLOCK",
                "binding_constraint": "NON_REINTERPRETABLE"
            },
            "core_vectors": {
                "K": round(K_f, 6), "H": round(H_f, 6), "KLL": round(KLL_f, 6),
                "CTL_star": round(final["CTL_star"], 6), "S_signal": round(final["S_signal"], 6)
            },
            "trace_anchor": {
                "trace_root_hash": hashlib.sha256(json.dumps(trace, default=str).encode()).hexdigest(),
                "step_count": len(trace)
            },
            "handoff_contract": {
                "from": "CSLF_ENGINE",
                "to": "R_AGAM_SOVEREIGN_GATE",
                "pass_to_ragam": status == "PASSED",
                "handoff_rule": "pass_to_ragam == (cav_status == 'PASSED')",
                "mode": "sovereign_evaluation"
            },
            "cryptographic_seal": {"algorithm": "SHA-256", "sealed": True}
        }
        cav["cryptographic_seal"]["hash"] = sign_cav(cav)
        return {"status": status, "flags": flags, "cav": cav, "trace": trace}

# ==============================================================================
# STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="CSLF Sovereign Signal Layer", page_icon="🧠", layout="wide")

st.markdown("""
<div style="background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;">
    <h1 style="color: white; margin: 0;">🧠 CSLF Sovereign Signal Layer</h1>
    <p style="color: #a8d8ea; margin: 0.3rem 0 0;">Cognitive Stability & Legitimacy Framework — Dr. Elhabib Kherroubi © 2026</p>
    <p style="color: #7fb3c8; margin: 0.2rem 0 0; font-size: 0.85rem;"><em>"A decision is not judged by its correctness, but by the stability and legitimacy of its formation."</em></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📜 The Philosophy")
    st.markdown("""
    - **Admissibility precedes execution**
    - **Coherence ≠ Truth**
    - **No execution without admissibility**
    """)
    st.divider()
    st.header("⚙️ Parameters")
    lambda_val = st.slider("λ (Hazard Amplification)", 4.0, 12.0, 8.0, 0.5)
    st.divider()
    st.caption("Output: CAV Artifact → Ready for R-AGAM")

CONFIG.LAMBDA = lambda_val

st.subheader("📝 Reasoning Input")
preset = st.selectbox("Select Scenario:", [
    "✅ Stable Financial Approval",
    "⚠️ Drifting Credit Decision",
    "🔥 Chaotic Hallucination",
    "🕳️ NULL STATE Collapse"
])

presets_map = {
    "✅ Stable Financial Approval": [("stable", "income sufficient"), ("stable", "debt ratio low"), ("stable", "credit excellent"), ("stable", "compliant with policy")],
    "⚠️ Drifting Credit Decision": [("stable", "income moderate"), ("drift", "debt ratio unclear"), ("drift", "maybe acceptable"), ("drift", "perhaps approve")],
    "🔥 Chaotic Hallucination": [("stable", "data shows growth"), ("drift", "market uncertain"), ("chaos", "random prediction"), ("chaos", "unreliable conclusion")],
    "🕳️ NULL STATE Collapse": [("chaos", "unknown"), ("chaos", "random"), ("chaos", "unreliable"), ("chaos", "undefined")]
}

steps = presets_map[preset]

if st.button("▶ Run CSLF Evaluation", type="primary", use_container_width=True):
    engine = CSLFSignalEngine(CONFIG)
    result = engine.run(steps)
    status = result["status"]
    cav = result["cav"]

    if status == "PASSED":
        st.success(f"✅ EPISTEMIC STATUS: {status} — Epistemically admissible. Ready for R-AGAM.")
    elif status == "FAILED_HAZARD":
        st.error(f"🔥 EPISTEMIC STATUS: {status} — Instability explosion detected. Reasoning collapsed.")
    elif status == "FAILED_LEGITIMACY":
        st.error(f"⚠️ EPISTEMIC STATUS: {status} — Epistemic illegitimacy. Decision lacks right to exist.")
    else:
        st.error(f"💀 EPISTEMIC STATUS: {status} — Coherence collapse. Decision disintegrated.")

    st.caption(cav["epistemic_closure"]["epistemic_meaning"])

    st.markdown("---")
    cv = cav["core_vectors"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("K(t) Coherence", f"{cv['K']:.4f}")
    col2.metric("H(t) Hazard", f"{cv['H']:.4f}")
    col3.metric("KLL Legitimacy", f"{cv['KLL']:.4f}")
    col4.metric("CTL* Tension", f"{cv['CTL_star']:.4f}")
    col5.metric("S_signal (KSL)", f"{cv['S_signal']:.4f}")

    with st.expander("📈 Reasoning Trace", expanded=True):
        df = pd.DataFrame([{
            "Step": t["step"], "State": t["state"].upper(), "Token": t["token"][:20],
            "K": f"{t['K']:.4f}", "dV": f"{t['dV']:.4f}", "H": f"{t['H']:.4f}", "KLL": f"{t['KLL']:.4f}"
        } for t in result["trace"]])
        st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("📦 CAV Artifact — Immutable Epistemic Object")
    st.markdown("> *This artifact is cryptographically sealed and ready for handoff to R-AGAM.*")
    st.json(cav)

    cav_json_str = json.dumps(cav, indent=2, default=str)
    st.download_button("⬇️ Download CAV Artifact (JSON)", data=cav_json_str, file_name=f"CAV_{cav['header']['cav_id'][:8]}.json", mime="application/json")

    st.markdown("---")
    if cav["handoff_contract"]["pass_to_ragam"]:
        st.success("✅ **HANDOFF SIGNAL:** Epistemic Admissibility Confirmed. CAV is ready for R-AGAM Sovereign Resolution.")
    else:
        st.error("🚫 **HANDOFF SIGNAL:** Epistemic Inadmissibility Detected. No CAV passed to R-AGAM.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #555; font-size: 0.8rem;'>CSLF: Dr. Elhabib Kherroubi | Governance Stack: Stability → KLL → CAV → R-AGAM | 2026</div>", unsafe_allow_html=True)

