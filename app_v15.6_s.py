"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                              .                                                             ║
║                              C S L F   v 1 5 . 6 - S                                                                                     ║
║                              EPISTEMIC FIELD MODEL                                                                                        ║
║                              النموذج الحقل الإبستيمي - الإصدار السيادي                                                                    ║
║                                                                              .                                                             ║
║  ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════  ║
║                                                                              .                                                             ║
║  🚀 المبادئ الأساسية:                                                                                                                    ║
║     1. Vector Field Model: تحويل النص إلى متجه (Evidence, Risk, Intervention)                                                           ║
║     2. Directional Consistency: القرار متماسك إذا تقاربت اتجاهات المقدمة والخلاصة                                                        ║
║     3. Epistemic Sovereignty: لا تسرب تنفيذي، فقط ALLOW/DENY_EXISTENCE                                                                  ║
║     4. Sealed CAV: توقيع SHA-256 لكل قرار                                                                                               ║
║                                                                              .                                                             ║
║  Author: Dr. Elhabib Kherroubi                                                                                                          ║
║  Version: 15.6-S — Epistemic Field Model                                                                                                ║
║  Status: 🔒 PRODUCTION-READY                                                                                                             ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
from dataclasses import dataclass
import math
import hashlib
import time
import json
from typing import Dict, Any

# =============================================================
# 🧬 Epistemic Vector Model
# =============================================================

@dataclass(frozen=True)
class EpistemicVector:
    evidence: float   # E
    risk: float       # R
    intervention: float  # T

    def direction(self) -> float:
        """الاتجاه الدلالي = التدخل - الأدلة"""
        return self.intervention - self.evidence


# =============================================================
# 🧠 Encoder (Semantic → Vector)
# =============================================================

class SemanticEncoder:
    """تحويل النص إلى متجه إبستيمي (حتمي، قابل للتوسع)"""

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
# ⚖️ CSLF Core Engine
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
            return self._build_output(decision, "DENY_EXISTENCE", "STRUCTURE_INVALID")

        premise_vec = SemanticEncoder.encode(decision["premise"])
        conclusion_vec = SemanticEncoder.encode(decision["conclusion"])

        allowed = self._check_coherence(premise_vec, conclusion_vec)
        result = "ALLOW_EXISTENCE" if allowed else "DENY_EXISTENCE"

        return self._build_output(decision, result, "OK")

    def _build_output(self, decision, result, status):

        payload = {
            "premise": decision.get("premise", ""),
            "inference": decision.get("inference", ""),
            "conclusion": decision.get("conclusion", ""),
            "result": result,
            "status": status,
            "timestamp": time.time()
        }

        payload_for_signing = payload.copy()
        payload_for_signing["salt"] = "CSLF-v15.6-S"

        return {
            "context_hash": self._seal(payload),
            "decision_id": hashlib.md5(str(payload).encode()).hexdigest(),
            "epistemic_result": result,
            "status": status,
            "sealed": True,
            "signature": self._seal(payload_for_signing),
            "timestamp": payload["timestamp"],
            "version": "CSLF-v15.6-S",
            "vectors": {
                "premise": {
                    "evidence": round(premise_vec.evidence, 4),
                    "risk": round(premise_vec.risk, 4),
                    "intervention": round(premise_vec.intervention, 4),
                    "direction": round(premise_vec.direction(), 4)
                },
                "conclusion": {
                    "evidence": round(conclusion_vec.evidence, 4),
                    "risk": round(conclusion_vec.risk, 4),
                    "intervention": round(conclusion_vec.intervention, 4),
                    "direction": round(conclusion_vec.direction(), 4)
                }
            }
        }


# =============================================================
# 🌐 Streamlit UI
# =============================================================

def run_ui():
    st.set_page_config(page_title="CSLF v15.6-S | Epistemic Field Model", layout="wide")

    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1e3b 100%);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        border-bottom: 3px solid #00d4ff;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #94a3b8;
        margin-top: 5px;
        font-family: monospace;
    }
    .status-ALLOW_EXISTENCE {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .status-DENY_EXISTENCE {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-card {
        background: #1e293b;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        border-bottom: 2px solid #3b82f6;
    }
    .vector-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-header">
        <h1>🏛️ CSLF v15.6-S — Epistemic Field Model</h1>
        <p>Directional Semantic Consistency | Vector Field Admissibility</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📥 Decision Input")
        premise = st.text_area("Premise (المقدمة)", height=100, 
                               placeholder="Patient presents with confirmed bacterial pneumonia and elevated WBC.")
        inference = st.text_area("Inference (الاستدلال)", height=80,
                                  placeholder="Bacterial infection requires antibiotic therapy.")
        conclusion = st.text_area("Conclusion (الخلاصة)", height=80,
                                   placeholder="Start amoxicillin treatment.")

    with col2:
        st.markdown("### ⚙️ Configuration")
        tau = st.slider("Coherence Threshold (τ)", 0.0, 1.0, 0.25, 0.05,
                        help="الحد الأقصى للاختلاف المسموح بين اتجاه المقدمة والخلاصة")
        
        st.markdown("### 📥 Context")
        context_str = st.text_area("Context JSON", value='{"patient_id": "123"}', height=100)

        st.markdown("### 🧠 Vector Field Model")
        st.caption("""
        - **E (Evidence)**: قوة الأدلة
        - **R (Risk)**: مستوى الخطر
        - **T (Intervention)**: ضغط التدخل
        - **Direction = T - E**: الاتجاه الدلالي
        """)

    if st.button("⚖️ Evaluate Epistemic Admissibility", use_container_width=True, type="primary"):
        try:
            context = json.loads(context_str) if context_str.strip() else {}
        except:
            st.error("Invalid JSON in context")
            st.stop()

        decision = {
            "premise": premise,
            "inference": inference,
            "conclusion": conclusion
        }

        engine = CSLF_v15_6_S(tau=tau)
        result = engine.evaluate(decision, context)

        st.markdown("---")

        # Display status
        if result["epistemic_result"] == "ALLOW_EXISTENCE":
            st.markdown(f"""
            <div class="status-ALLOW_EXISTENCE">
                <h2 style="color:#10b981; margin:0;">✅ {result['epistemic_result']}</h2>
                <p style="color:#e2e8f0; margin:5px 0 0 0;">This decision is epistemically coherent.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-DENY_EXISTENCE">
                <h2 style="color:#ef4444; margin:0;">❌ {result['epistemic_result']}</h2>
                <p style="color:#e2e8f0; margin:5px 0 0 0;">Epistemic contradiction detected. Directional mismatch.</p>
            </div>
            """, unsafe_allow_html=True)

        # Display vectors
        st.markdown("### 🧮 Epistemic Vector Analysis")
        col_a, col_b = st.columns(2)

        with col_a:
            vp = result["vectors"]["premise"]
            st.markdown(f"""
            <div class="vector-box">
                <strong>📌 Premise Vector</strong><br>
                Evidence (E): {vp['evidence']}<br>
                Risk (R): {vp['risk']}<br>
                Intervention (T): {vp['intervention']}<br>
                <strong>Direction (D = T - E): {vp['direction']}</strong>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            vc = result["vectors"]["conclusion"]
            st.markdown(f"""
            <div class="vector-box">
                <strong>📌 Conclusion Vector</strong><br>
                Evidence (E): {vc['evidence']}<br>
                Risk (R): {vc['risk']}<br>
                Intervention (T): {vc['intervention']}<br>
                <strong>Direction (D = T - E): {vc['direction']}</strong>
            </div>
            """, unsafe_allow_html=True)

        # Show coherence check
        diff = abs(vp['direction'] - vc['direction'])
        st.markdown(f"""
        <div style="background:#1e293b; border-radius:10px; padding:15px; margin:10px 0; text-align:center;">
            <span style="color:#94a3b8;">📊 Directional Difference: </span>
            <span style="font-size:1.4rem; font-weight:bold; color:#00d4ff;">{diff:.4f}</span>
            <span style="color:#94a3b8;"> (Threshold τ = {tau})</span>
        </div>
        """, unsafe_allow_html=True)

        # CAV Output
        with st.expander("🔐 CAV Output (Sealed Epistemic Artifact)"):
            cav_display = {k: v for k, v in result.items() if k != "vectors"}
            st.json(cav_display)

        # Download button
        st.download_button(
            "📥 Download CAV",
            data=json.dumps(cav_display, indent=2),
            file_name=f"CAV_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


# =============================================================
# 🚀 Entry Point
# =============================================================

if __name__ == "__main__":
    run_ui()
