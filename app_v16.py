# =============================================================
# 🏛️ CSLF v16 — Epistemic Dynamics Engine
# Author: Dr. Elhabib Kherroubi
# Core Principle: Epistemic → Action Mapping (NOT Direction Matching)
# =============================================================

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
    evidence: float        # نفي الحالة
    risk: float            # خطر
    uncertainty: float     # عدم يقين
    intervention: float    # تدخل

# =============================================================
# 🧠 Semantic Encoder
# =============================================================

class SemanticEncoder:

    evidence_keywords = [
        "no evidence", "normal", "benign", "stable",
        "no infection", "unremarkable", "negative",
        "no signs", "absence", "without"
    ]

    risk_keywords = [
        "infection", "fever", "elevated", "abnormal",
        "severe", "critical", "high", "positive",
        "confirmed", "bacterial", "pneumonia", "wbc"
    ]

    uncertainty_keywords = [
        "possible", "suspected", "cannot rule out",
        "likely", "maybe", "consider", "unclear"
    ]

    intervention_keywords = [
        "treat", "administer", "start", "give",
        "antibiotic", "therapy", "medication",
        "drug", "prescribe", "intervention", "treatment"
    ]

    @staticmethod
    def encode(text: str) -> EpistemicVector:
        t = text.lower()

        e = sum(1 for k in SemanticEncoder.evidence_keywords if k in t)
        r = sum(1 for k in SemanticEncoder.risk_keywords if k in t)
        u = sum(1 for k in SemanticEncoder.uncertainty_keywords if k in t)
        i = sum(1 for k in SemanticEncoder.intervention_keywords if k in t)

        norm = max(e + r + u + i, 1)

        return EpistemicVector(
            evidence=e / norm,
            risk=r / norm,
            uncertainty=u / norm,
            intervention=i / norm
        )

# =============================================================
# ⚖️ CSLF Core Engine (v16)
# =============================================================

class CSLF_v16:

    def __init__(self, tau: float = 0.25):
        self.tau = tau

    # -------------------------
    # Structure Check
    # -------------------------
    def _check_structure(self, decision: Dict[str, str]) -> bool:
        required = ["premise", "inference", "conclusion"]
        return all(
            k in decision and isinstance(decision[k], str) and decision[k].strip()
            for k in required
        )

    # -------------------------
    # Expected Action Model
    # -------------------------
    def _expected_action(self, vp: EpistemicVector) -> float:
        """
        A_exp = R + 0.5U - E
        """
        return vp.risk + 0.5 * vp.uncertainty - vp.evidence

    # -------------------------
    # Coherence Check
    # -------------------------
    def _check_coherence(self, vp: EpistemicVector, vc: EpistemicVector) -> bool:
        expected = self._expected_action(vp)
        actual = vc.intervention
        return abs(expected - actual) <= self.tau

    # -------------------------
    # Sealing
    # -------------------------
    def _seal(self, payload: Dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    # -------------------------
    # Evaluation
    # -------------------------
    def evaluate(self, decision: Dict[str, str], context: Dict[str, Any] = None):

        if not self._check_structure(decision):
            return self._build_output(decision, None, None, "DENY_EXISTENCE", "STRUCTURE_INVALID")

        premise_vec = SemanticEncoder.encode(decision["premise"])
        conclusion_vec = SemanticEncoder.encode(decision["conclusion"])

        allowed = self._check_coherence(premise_vec, conclusion_vec)
        result = "ALLOW_EXISTENCE" if allowed else "DENY_EXISTENCE"

        return self._build_output(decision, premise_vec, conclusion_vec, result, "OK")

    # -------------------------
    # Output Builder
    # -------------------------
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

        payload_signed = payload.copy()
        payload_signed["salt"] = "CSLF-v16"

        output = {
            "context_hash": self._seal(payload),
            "decision_id": hashlib.md5(str(payload).encode()).hexdigest(),
            "epistemic_result": result,
            "status": status,
            "sealed": True,
            "signature": self._seal(payload_signed),
            "timestamp": timestamp,
            "version": "CSLF-v16"
        }

        # إضافة المتجهات فقط إذا كانت موجودة
        if vp and vc:
            output["vectors"] = {
                "premise": {
                    "evidence": round(vp.evidence, 4),
                    "risk": round(vp.risk, 4),
                    "uncertainty": round(vp.uncertainty, 4),
                    "expected_action": round(self._expected_action(vp), 4)
                },
                "conclusion": {
                    "intervention": round(vc.intervention, 4)
                }
            }

        return output

# =============================================================
# 🌐 Streamlit UI
# =============================================================

def run_ui():

    st.set_page_config(page_title="CSLF v16 — Epistemic Dynamics Engine", layout="wide")

    st.title("🏛️ CSLF v16 — Epistemic Dynamics Engine")
    st.caption("Epistemic → Action Mapping | Sovereign Decision Gate")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📥 Decision Input")
        premise = st.text_area("Premise", height=120)
        inference = st.text_area("Inference", height=100)
        conclusion = st.text_area("Conclusion", height=100)

    with col2:
        st.subheader("⚙️ Configuration")
        tau = st.slider("Coherence Threshold τ", 0.0, 1.0, 0.25, 0.05)

        st.subheader("📥 Context")
        context_str = st.text_area("Context JSON", value='{"patient_id": "123"}')

    if st.button("⚖️ Evaluate", use_container_width=True):

        try:
            context = json.loads(context_str) if context_str.strip() else {}
        except:
            st.error("Invalid JSON")
            st.stop()

        decision = {
            "premise": premise,
            "inference": inference,
            "conclusion": conclusion
        }

        engine = CSLF_v16(tau=tau)
        result = engine.evaluate(decision, context)

        st.markdown("---")

        if result["epistemic_result"] == "ALLOW_EXISTENCE":
            st.success("✅ ALLOW_EXISTENCE")
        else:
            st.error("❌ DENY_EXISTENCE")

        if "vectors" in result:
            st.subheader("🧮 Epistemic Dynamics")
            st.json(result["vectors"])

        with st.expander("🔐 CAV Output"):
            cav = {k: v for k, v in result.items() if k != "vectors"}
            st.json(cav)

# =============================================================
# 🚀 Entry Point
# =============================================================

if __name__ == "__main__":
    run_ui()
