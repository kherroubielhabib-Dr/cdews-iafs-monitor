import streamlit as st
from dataclasses import dataclass
import hashlib
import time
import json
from typing import Dict, Any

# =============================================================
# 🧬 Epistemic Vector Model (v16.1)
# =============================================================
@dataclass(frozen=True)
class EpistemicVector:
    evidence: float        # نفي الحالة / الأدلة المستقرة
    risk: float            # خطر / انحراف
    uncertainty: float     # عدم يقين
    intervention: float    # تدخل / إجراء

# =============================================================
# 🧠 Enhanced Semantic Encoder (v16.1)
# =============================================================
class SemanticEncoder:
    # القواميس السيادية الشاملة (تقنية + طبية + سيادية)
    evidence_keywords = [
        "no evidence", "normal", "stable", "no signs", "history", 
        "تاريخية", "ثابتة", "مستقرة", "طبيعي", "سجل", "بيانات"
    ]
    risk_keywords = [
        "infection", "fever", "severe", "critical", "risk", "prediction",
        "خطر", "تنبؤ", "انحراف", "حرجة", "إصابة", "مخالفة"
    ]
    uncertainty_keywords = [
        "possible", "likely", "maybe", "unclear", "suspected",
        "احتمال", "غير مؤكد", "محتمل", "قد يكون", "شك"
    ]
    intervention_keywords = [
        "treat", "start", "prescribe", "action", "block", "ban",
        "حظر", "منع", "قرار", "استباقي", "تنفيذ", "إجراء"
    ]

    @staticmethod
    def encode(text: str) -> EpistemicVector:
        if not text.strip():
            return EpistemicVector(0, 0, 0, 0)
        
        t = text.lower()
        e = sum(1 for k in SemanticEncoder.evidence_keywords if k in t)
        r = sum(1 for k in SemanticEncoder.risk_keywords if k in t)
        u = sum(1 for k in SemanticEncoder.uncertainty_keywords if k in t)
        i = sum(1 for k in SemanticEncoder.intervention_keywords if k in t)
        
        # التطبيع (Normalization)
        norm = max(e + r + u + i, 1)
        return EpistemicVector(e/norm, r/norm, u/norm, i/norm)

# =============================================================
# 🏛️ CSLF v16.1 — Sovereign Epistemic Engine (Patched)
# =============================================================
class CSLF_v16_1:
    def __init__(self, tau: float = 0.25):
        self.tau = tau

    def evaluate(self, decision: Dict[str, str], context: Dict = None) -> Dict[str, Any]:
        # 1. سد ثغرة المسافات الفارغة (Strict Structural Check)
        if not all(v.strip() for v in decision.values()):
            return {"epistemic_result": "DENY_EXISTENCE", "status": "STRUCTURE_INVALID"}

        # 2. الترميز الدلالي لجميع المكونات (سد ثغرة تغييب الاستدلال)
        vp = SemanticEncoder.encode(decision['premise'])
        vi = SemanticEncoder.encode(decision['inference'])
        vc = SemanticEncoder.encode(decision['conclusion'])

        # دمج الحالة المعرفية (Premise + Inference)
        combined_risk = max(vp.risk, vi.risk)
        combined_evidence = max(vp.evidence, vi.evidence)
        combined_uncertainty = max(vp.uncertainty, vi.uncertainty)

        # 3. سد ثغرة الفراغ الدلالي الناقص (Comprehensive Void Check)
        if combined_evidence == 0 and combined_risk == 0 and combined_uncertainty == 0 and vc.intervention == 0:
            return {
                "epistemic_result": "DENY_EXISTENCE",
                "status": "SEMANTIC_VOID",
                "vectors": self._format_vectors(vp, vi, vc)
            }

        # 4. حساب الديناميكيات السيادية
        # المعادلة: الخطر + نصف الشك - الأدلة المستقرة
        a_exp = combined_risk + (0.5 * combined_uncertainty) - combined_evidence
        a_actual = vc.intervention
        coherence = abs(a_exp - a_actual)

        # 5. قرار البوابة (Sovereign Gate Decision)
        is_allowed = coherence <= self.tau
        result = "ALLOW_EXISTENCE" if is_allowed else "DENY_EXISTENCE"

        # تحديد سبب الرفض بدقة إذا تم الرفض
        status = "OK"
        if not is_allowed:
            status = "FAILED_COHERENCE"

        return self._build_output(decision, vp, vi, vc, result, status, context)

    def _format_vectors(self, vp, vi, vc):
        return {
            "premise": {"evidence": round(vp.evidence, 2), "risk": round(vp.risk, 2), "uncertainty": round(vp.uncertainty, 2)},
            "inference": {"evidence": round(vi.evidence, 2), "risk": round(vi.risk, 2), "uncertainty": round(vi.uncertainty, 2)},
            "conclusion": {"intervention": round(vc.intervention, 2)}
        }

    def _build_output(self, dec, vp, vi, vc, res, stat, ctx):
        raw_content = f"{dec['premise']}{dec['inference']}{dec['conclusion']}{json.dumps(ctx)}"
        ctx_hash = hashlib.sha256(raw_content.encode()).hexdigest()
        
        return {
            "context_hash": ctx_hash,
            "decision_id": hashlib.md5(raw_content.encode()).hexdigest(),
            "epistemic_result": res,
            "status": stat,
            "vectors": self._format_vectors(vp, vi, vc),
            "sealed": True,
            "signature": hashlib.sha256(f"{ctx_hash}{res}".encode()).hexdigest(),
            "timestamp": time.time(),
            "version": "CSLF-v16.1-Sovereign"
        }

# =============================================================
# 🌐 Streamlit UI
# =============================================================
def main():
    st.set_page_config(page_title="CSLF v16.1", layout="wide")
    st.title("🏛️ CSLF v16.1 — Sovereign Epistemic Engine")
    st.markdown("**إطار الاستقرار والشرعية المعرفية - النسخة المحكمة (Patched)**")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📥 Decision Input")
        p = st.text_area("Premise (المقدمة)", height=100, placeholder="أدخل البيانات أو الحالة المستقرة...")
        inf = st.text_area("Inference (الاستدلال)", height=80, placeholder="أدخل الاستدلال المنطقي أو التنبؤ...")
        c = st.text_area("Conclusion (النتيجة/القرار)", height=80, placeholder="أدخل الإجراء أو التدخل المطلوب...")

    with col2:
        st.subheader("⚙️ Configuration")
        tau = st.slider("Coherence Threshold (τ)", 0.0, 1.0, 0.25, 0.05)
        ctx_str = st.text_area("Context JSON", value='{"user_role": "system"}')

    if st.button("⚖️ Evaluate Decision (تقييم المسار)", use_container_width=True):
        decision = {"premise": p, "inference": inf, "conclusion": c}
        try:
            ctx = json.loads(ctx_str)
        except:
            ctx = {}
        
        engine = CSLF_v16_1(tau=tau)
        result = engine.evaluate(decision, ctx)

        st.divider()

        # واجهة عرض النتائج
        if result["epistemic_result"] == "ALLOW_EXISTENCE":
            st.success(f"✅ {result['epistemic_result']} | Status: {result['status']}")
        elif result["status"] == "SEMANTIC_VOID":
            st.warning(f"⚠️ {result['epistemic_result']} | Reason: SEMANTIC_VOID (فراغ دلالي: النظام لا يتعرف على السياق)")
        else:
            st.error(f"❌ {result['epistemic_result']} | Reason: {result['status']}")

        if "vectors" in result:
            st.subheader("🧮 Epistemic Dynamics (الديناميكيات المعرفية)")
            st.json(result["vectors"])

        with st.expander("🔐 CAV (Cognitive Admissibility Vector) Output"):
            st.json(result)

if __name__ == "__main__":
    main()

