import streamlit as st
from dataclasses import dataclass
import hashlib
import time
import json
from typing import Dict, Any

# =============================================================
# 🧬 Epistemic Vector Model
# =============================================================

@dataclass(frozen=True)
class EpistemicVector:
    evidence: float
    risk: float
    intervention: float

    def direction(self) -> float:
        return self.intervention - self.evidence


# =============================================================
# 🧠 Semantic Encoder
# =============================================================

class SemanticEncoder:

    evidence_keywords = [
        "no evidence", "normal", "benign", "stable",
        "no infection", "unremarkable", "negative", "no signs",
        "absence", "without", "clear", "healthy"
    ]

    risk_keywords = [
        "infection", "fever", "elevated", "abnormal",
        "severe", "critical", "high", "positive",
        "confirmed", "bacterial", "pneumonia", "wbc"
    ]

    intervention_keywords = [
        "treat", "administer", "start", "give",
        "antibiotic", "therapy", "medication", "drug",
        "prescribe", "intervention", "treatment"
    ]

    @staticmethod
    def encode(text: str) -> EpistemicVector:
        t = text.lower()

        e = sum(1 for k in SemanticEncoder.evidence_keywords if k in t)
        r = sum(1 for k in SemanticEncoder.risk_keywords if k in t)
        i = sum(1 for k in SemanticEncoder.intervention_keywords if k in t)

        norm = max(e + r + i, 1)

        return EpistemicVector(
            evidence=e / norm,
            risk=r / norm,
            intervention=i / norm
        )


# =============================================================
# ⚖️ CSLF Engine
# =============================================================

class CSLF_v15_6_S:

    def __init__(self, tau: float = 0.25):
        self.tau = tau

    def _check_structure(self, decision: Dict[str, str]) -> bool:
        required = ["premise", "inference", "conclusion"]
        return all(
            k in decision and isinstance(decision[k], str) and decision[k].strip()
            for k in required
        )

    def _check_coherence(self, vp: EpistemicVector, vc: EpistemicVector) -> bool:
        dp = vp.direction()
        dc = vc.direction()
        return abs(dp - dc) <= self.tau

    def _seal(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def evaluate(self, decision: Dict[str, str], context: Dict[str, Any] = None):

        if not self._check_structure(decision):
            return self._build_output(decision, None, None, "DENY_EXISTENCE", "STRUCTURE_INVALID")

        premise_vec = SemanticEncoder.encode(decision["premise"])
        conclusion_vec = SemanticEncoder.encode(decision["conclusion"])

        allowed = self._check_coherence(premise_vec, conclusion_vec)
        result = "ALLOW_EXISTENCE" if allowed else "DENY_EXISTENCE"

        return self._build_output(decision, premise_vec, conclusion_vec, result, "OK")

    def _build_output(self, decision, vp, vc, result, status):

        timestamp = time.time()

        payload = {
            "premise": decision.get("premise", ""),
            "inference": decision.get("inference", ""),
            "conclusion": decision.get("conclusion", ""),
            "result": result,
            "status": status,
            "timestamp": timestamp
        }

        payload_for_signing = payload.copy()
        payload_for_signing["salt"] = "CSLF-v15.6-S"

        output = {
            "context_hash": self._seal(payload),
            "decision_id": hashlib.md5(json.dumps(payload).encode()).hexdigest(),
            "epistemic_result": result,
            "status": status,
            "sealed": True,
            "signature": self._seal(payload_for_signing),
            "timestamp": timestamp,
            "version": "CSLF-v15.6-S"
        }

        # ✅ فقط إذا كانت vectors موجودة
        if vp and vc:
            output["vectors"] = {
                "premise": {
                    "evidence": round(vp.evidence, 4),
                    "risk": round(vp.risk, 4),
                    "intervention": round(vp.intervention, 4),
                    "direction": round(vp.direction(), 4)
                },
                "conclusion": {
                    "evidence": round(vc.evidence, 4),
                    "risk": round(vc.risk, 4),
                    "intervention": round(vc.intervention, 4),
                    "direction": round(vc.direction(), 4)
                }
            }

        return output


# =============================================================
# 🌐 Streamlit UI
# =============================================================

def run_ui():

    st.set_page_config(page_title="CSLF v15.6-S", layout="wide")

    st.title("🏛️ CSLF v15.6-S — Epistemic Field Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        premise = st.text_area("Premise", height=100)
        inference = st.text_area("Inference", height=80)
        conclusion = st.text_area("Conclusion", height=80)

    with col2:
        tau = st.slider("τ", 0.0, 1.0, 0.25)
        context_str = st.text_area("Context", value='{"patient_id":"123"}')

    if st.button("Evaluate"):

        try:
            context = json.loads(context_str)
        except:
            st.error("Invalid JSON")
            return

        decision = {
            "premise": premise,
            "inference": inference,
            "conclusion": conclusion
        }

        engine = CSLF_v15_6_S(tau)
        result = engine.evaluate(decision, context)

        st.json(result)


# =============================================================
# 🚀 Entry
# =============================================================

if __name__ == "__main__":
    run_ui()
