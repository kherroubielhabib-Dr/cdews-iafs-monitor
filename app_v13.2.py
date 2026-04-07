import streamlit as st
import math
from dataclasses import dataclass

# =========================
# CONFIG
# =========================
EPSILON = 1e-9
LAMBDA = 2.0
THETA_K = 0.6
THETA_H = 1.5

# =========================
# DATA STRUCTURES
# =========================
@dataclass
class EpistemicState:
    K: float
    I: float

@dataclass
class KLLComponents:
    SA: float
    CI: float
    SV: float
    EG: float
    w: float = 1.0

# =========================
# ENGINE
# =========================
class CSLFEngine:
    def __init__(self):
        self.prev_V = None
        self.prev_I = None

    def compute_V(self, K):
        return -math.log(K + EPSILON)

    def compute_dV(self, V):
        if self.prev_V is None:
            self.prev_V = V
            return 0.0
        dV = max(0, V - self.prev_V)
        self.prev_V = V
        return dV

    def compute_dI(self, I):
        if self.prev_I is None:
            self.prev_I = I
            return 0.0
        dI = abs(I - self.prev_I)
        self.prev_I = I
        return dI

    def compute_H(self, dI, dV):
        return dI * math.exp(LAMBDA * dV)

    def compute_KLL(self, kll):
        components = [kll.SA, kll.CI, kll.SV, kll.EG]

        for x in components:
            if x <= 0:
                return 0.0

        product = kll.SA * kll.CI * kll.SV * kll.EG
        return product ** kll.w

    def evaluate(self, state, kll):
        V = self.compute_V(state.K)
        dV = self.compute_dV(V)
        dI = self.compute_dI(state.I)
        H = self.compute_H(dI, dV)

        KLL_value = self.compute_KLL(kll)

        if KLL_value == 0:
            return "FAILED_LEGITIMACY", H, KLL_value

        if H > THETA_H:
            return "FAILED_HAZARD", H, KLL_value

        if state.K < THETA_K:
            return "FAILED_COHERENCE", H, KLL_value

        return "CAV_ACCEPTED", H, KLL_value


# =========================
# UI
# =========================
st.set_page_config(page_title="CSLF v13.2 Demo", layout="centered")

st.title("🧠 CSLF v13.2 — Epistemic Admissibility Engine")
st.markdown("**هذا النظام يحسم أهلية القرار قبل التنفيذ — دون أي تدخل تنفيذي**")

st.header("📊 إدخال الحالة المعرفية")

K = st.slider("Cognitive Coherence K", 0.0, 1.0, 0.75)
I = st.slider("Information State I", 0.0, 1.0, 0.4)

st.header("🧩 مكونات الشرعية (KLL)")

SA = st.slider("Structural Adequacy (SA)", 0.0, 1.0, 1.0)
CI = st.slider("Consistency Integrity (CI)", 0.0, 1.0, 1.0)
SV = st.slider("Semantic Validity (SV)", 0.0, 1.0, 1.0)
EG = st.slider("Evidence Grounding (EG)", 0.0, 1.0, 1.0)

if st.button("تشغيل التقييم"):

    engine = CSLFEngine()

    state = EpistemicState(K=K, I=I)
    kll = KLLComponents(SA=SA, CI=CI, SV=SV, EG=EG)

    result, H, KLL_value = engine.evaluate(state, kll)

    st.subheader("📌 النتيجة")

    st.write(f"**H (Hazard):** {H:.4f}")
    st.write(f"**KLL:** {KLL_value:.4f}")

    if result == "CAV_ACCEPTED":
        st.success("✅ تم إنتاج CAV (قرار مؤهل للوجود)")
    elif result == "FAILED_LEGITIMACY":
        st.error("❌ FAILED_LEGITIMACY — انهيار الشرعية")
    elif result == "FAILED_HAZARD":
        st.error("⚠️ FAILED_HAZARD — خطر غير مقبول")
    elif result == "FAILED_COHERENCE":
        st.error("❌ FAILED_COHERENCE — تماسك غير كاف")

st.markdown("---")
st.caption("CSLF does NOT execute decisions — it only determines admissibility.")
