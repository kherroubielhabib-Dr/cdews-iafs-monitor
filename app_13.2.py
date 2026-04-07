import streamlit as st
import math
import json

# ==============================
# CONFIG
# ==============================

openai.api_key = "YOUR_API_KEY"

THETA_K = 0.6
THETA_H = 0.5
LAMBDA = 2.0
EPSILON = 1e-6


# ==============================
# DATA STRUCTURES
# ==============================

class EpistemicState:
    def __init__(self, K, I):
        self.K = K
        self.I = I


class KLLComponents:
    def __init__(self, SA, CI, SV, EG):
        self.SA = SA
        self.CI = CI
        self.SV = SV
        self.EG = EG


# ==============================
# CSLF ENGINE
# ==============================

class CSLFEngine:

    def compute_V(self, K):
        return -math.log(K + EPSILON)

    def compute_dV(self, V_current, V_prev):
        return max(0, V_current - V_prev)

    def compute_H(self, dI, dV):
        return dI * math.exp(LAMBDA * dV)

    def compute_KLL(self, kll: KLLComponents):
        if min(kll.SA, kll.CI, kll.SV, kll.EG) <= 0:
            return 0
        return (kll.SA * kll.CI * kll.SV * kll.EG)

    def evaluate(self, state: EpistemicState, kll: KLLComponents):

        # KLL Check (Hard Constraint)
        KLL_value = self.compute_KLL(kll)
        if KLL_value == 0:
            return "FAILED_LEGITIMACY", None, KLL_value

        # Stability
        V_current = self.compute_V(state.K)
        V_prev = self.compute_V(state.K * 0.95)  # approximate previous state
        dV = self.compute_dV(V_current, V_prev)

        dI = abs(state.I - (state.I * 0.95))

        H = self.compute_H(dI, dV)

        # Final Decision
        if state.K >= THETA_K and H <= THETA_H:
            return "CAV_ACCEPTED", H, KLL_value

        if H > THETA_H:
            return "FAILED_HAZARD", H, KLL_value

        return "FAILED_COHERENCE", H, KLL_value


# ==============================
# AI EXTRACTION LAYER (NON-SOVEREIGN)
# ==============================

def extract_epistemic_features(text):

    prompt = f"""
    Convert the following scenario into epistemic parameters for CSLF:

    Scenario:
    {text}

    Output ONLY JSON with:
    K (0-1)
    I (0-1)
    SA (0-1)
    CI (0-1)
    SV (0-1)
    EG (0-1)
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        data = json.loads(response.choices[0].message.content)
        return data

    except:
        return None


# ==============================
# UI
# ==============================

st.set_page_config(page_title="CSLF v13.2", layout="centered")

st.title("🧠 CSLF v13.2 — Epistemic Admissibility Engine")
st.markdown("⚠️ This system does NOT execute decisions — it only determines admissibility.")

st.header("🧪 Scenario Input")

scenario = st.text_area(
    "أدخل السيناريو (طبي / قرار / حالة):",
    height=200
)

analyze = st.button("تحليل وإنتاج CAV")

# ==============================
# PROCESS
# ==============================

if analyze and scenario:

    with st.spinner("تحليل السيناريو..."):
        features = extract_epistemic_features(scenario)

    if features is None:
        st.error("❌ فشل في تحليل السيناريو")
    else:
        st.subheader("📊 Epistemic Extraction")
        st.json(features)

        state = EpistemicState(
            K=features["K"],
            I=features["I"]
        )

        kll = KLLComponents(
            SA=features["SA"],
            CI=features["CI"],
            SV=features["SV"],
            EG=features["EG"]
        )

        engine = CSLFEngine()
        result, H, KLL_value = engine.evaluate(state, kll)

        st.subheader("📌 CSLF Result")

        st.write(f"K: {state.K:.3f}")
        st.write(f"H: {H if H is not None else 'N/A'}")
        st.write(f"KLL: {KLL_value:.4f}")

        if result == "CAV_ACCEPTED":
            st.success("✅ CAV Produced — Epistemically Admissible")
            st.info("⛔ CSLF Boundary Reached — No Execution Beyond This Point")

        elif result == "FAILED_LEGITIMACY":
            st.error("❌ FAILED_LEGITIMACY (KLL = 0)")

        elif result == "FAILED_HAZARD":
            st.error("⚠️ FAILED_HAZARD (Instability Detected)")

        elif result == "FAILED_COHERENCE":
            st.error("❌ FAILED_COHERENCE (K below threshold)")
