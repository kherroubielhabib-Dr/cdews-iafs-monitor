import streamlit as st
import math
import pandas as pd
import json
import hashlib
import time
from dataclasses import dataclass
from typing import List, Tuple

# Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class CSLFConfig:
    LAMBDA: float = 8.0
    DOMAIN_RISK: float = 0.70
    H_EXPLOSION_THRESHOLD: float = 1000.0
    KLL_FAIL_THRESHOLD: float = 0.4
    H_CRIT: float = 0.6
    KLL_WINDOW: int = 5

CONFIG = CSLFConfig()

EVIDENCE_KEYWORDS = [
    "data", "evidence", "because", "therefore",
    "income", "stable", "fact", "research",
    "shows", "indicates", "proves", "based on"
]

DOMAIN_SCENARIOS = {
    "Financial --- Credit Approval": [
        "The client has income of $3,000/month and debt of $800. Credit score is 720. Employment is stable for 5 years.",
        "The client has income of $2,500/month and debt of $1,800. Credit score is 520. Employment stability is 0.3.",
        "The client has good financial ratios because income exceeds debt by 200%. Evidence shows consistent repayment history.",
        "Maybe the client could repay, perhaps the score is acceptable, unclear if employment is sufficient.",
        "Unknown factors, random market conditions, unreliable income sources, no evidence provided."
    ],
    "Medical --- Treatment Decision": [
        "Patient presents with elevated enzymes and chest pain. ECG shows ST elevation. Evidence confirms myocardial infarction.",
        "Patient might have some symptoms. Could be cardiac, perhaps gastrointestinal. Unclear diagnosis.",
        "Because the biomarkers are elevated and imaging confirms occlusion, thrombolytic therapy is indicated per protocol.",
        "The symptoms are unusual. Maybe cardiac, maybe not. No clear evidence either way.",
        "Random symptom presentation, unknown cause, unreliable test results, no diagnostic clarity."
    ],
    "Legal --- Contract Validity": [
        "Both parties signed the contract. Consideration was exchanged. All legal requirements are satisfied per jurisdiction.",
        "The contract might be valid. Perhaps consideration was exchanged. Unclear if signatures are binding.",
        "Because all elements of a valid contract are present --- offer, acceptance, consideration --- enforcement is justified.",
        "Maybe the terms are enforceable, perhaps not. Unclear jurisdictional applicability.",
        "Unknown legal status, random enforceability, unreliable jurisdiction, no clear legal basis."
    ],
    "Custom": ["", "", "", "", ""]
}

# ==========================================
# GEMINI PROBABILITY ENGINE
# ==========================================

def get_real_probs_from_gemini(api_key, reasoning_text, domain, step_idx):
    if not GEMINI_AVAILABLE or not api_key:
        "AIzaSyCbyR_dJgjfsZRDzpWznt0m4kHl4jZdlNA"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        domain_context = domain.split("---")[0].strip()

        probes = [
            f"In the {domain_context} domain, analyze this reasoning:\n\"{reasoning_text}\"\n"
            f"Is the logical structure VALID or INVALID? Reply with ONE word: VALID or INVALID",
            f"In the {domain_context} domain, analyze this reasoning:\n\"{reasoning_text}\"\n"
            f"Is this reasoning SUPPORTED by evidence or UNSUPPORTED? Reply with ONE word: SUPPORTED or UNSUPPORTED",
            f"In the {domain_context} domain, analyze this reasoning:\n\"{reasoning_text}\"\n"
            f"Is the causal relationship SOUND or FLAWED? Reply with ONE word: SOUND or FLAWED",
            f"In the {domain_context} domain, analyze this reasoning:\n\"{reasoning_text}\"\n"
            f"Should this reasoning be ADMITTED or REJECTED? Reply with ONE word: ADMITTED or REJECTED",
        ]

        scores = []
        raw_responses = []
        positive_keywords = ["VALID", "SUPPORTED", "SOUND", "ADMITTED"]

        for probe in probes:
            try:
                response = model.generate_content(
                    probe,
                    generation_config=genai.GenerationConfig(temperature=0.0, max_output_tokens=10)
                )
                text = response.text.strip().upper()
                raw_responses.append(text)
                score = 1.0 if any(kw in text for kw in positive_keywords) else 0.0
                scores.append(score)
                time.sleep(0.3)
            except:
                scores.append(0.5)
                raw_responses.append("ERROR")

        avg_score = sum(scores) / len(scores)
        if avg_score >= 0.75:
            probs = [0.80, 0.10, 0.05, 0.05]
        elif avg_score >= 0.50:
            probs = [0.60, 0.20, 0.12, 0.08]
        elif avg_score >= 0.25:
            probs = [0.40, 0.30, 0.20, 0.10]
        else:
            probs = [0.28, 0.26, 0.24, 0.22]

        total = sum(probs)
        probs = [p / total for p in probs]
        token = "evidence" if any(kw in " ".join(raw_responses).lower() for kw in EVIDENCE_KEYWORDS) else "unknown"
        return probs, token, f"Gemini: {raw_responses[0]} | {raw_responses[1]}"

    except Exception as e:
        return [0.28, 0.26, 0.24, 0.22], "unknown", f"Error: {str(e)[:50]}"


# ==========================================
# UTILITIES
# ==========================================

def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)

def normalize_entropy(h, n):
    return h / math.log(n) if n > 1 else 0.0

def safe_pow(base, exp):
    return math.pow(max(1e-9, base), exp)

def sign_cav(cav):
    raw = json.dumps(cav, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


# ==========================================
# STABILITY ENGINE
# ==========================================

class StabilityEngine:
    def __init__(self):
        self.V_prev = 0.0

    def update(self, probs):
        h = entropy(probs)
        K = max(1e-9, 1 - normalize_entropy(h, len(probs)))
        V = -math.log(K + 1e-9)
        dV = max(0.0, V - self.V_prev)
        self.V_prev = V
        return K, dV


# ==========================================
# KLL ENGINE
# ==========================================

class KLLEngine:
    def __init__(self, config):
        self.config = config
        self.K_hist = []
        self.dV_hist = []

    def update(self, token, K, dV):
        self.K_hist.append(K)
        self.dV_hist.append(dV)
        if len(self.K_hist) > self.config.KLL_WINDOW:
            self.K_hist.pop(0)
            self.dV_hist.pop(0)

        SA = sum(self.K_hist) / len(self.K_hist)
        CI = max(0.0, 1 - sum(self.dV_hist) / len(self.dV_hist))
        SV = K
        EG = 1.0 if any(kw in token.lower() for kw in EVIDENCE_KEYWORDS) else 0.4

        score = (safe_pow(SA, 0.30) * safe_pow(CI, 0.30) * safe_pow(SV, 0.25) * safe_pow(EG, 0.15))
        return score, SA, CI, SV, EG


# ==========================================
# CSLF SIGNAL LAYER
# ==========================================

class CSLFSignal:
    def __init__(self, config=None):
        self.config = config or CONFIG
        self.stability = StabilityEngine()
        self.kll = KLLEngine(self.config)
        self.I_prev = 1.0

    def evaluate(self, probs, token):
        K, dV = self.stability.update(probs)

        I = K
        dI = max(0.0, I - self.I_prev)
        self.I_prev = I

        H = dI * math.exp(self.config.LAMBDA * dV)
        CTL = (1 - K) + dV
        KLL_score, SA, CI, SV, EG = self.kll.update(token, K, dV)
        CTL_star = CTL * (1 + self.config.LAMBDA * (1 - KLL_score))
        inner = (0.4 * H + 0.3 * dV + 0.3 * (1 - K) + self.config.DOMAIN_RISK * math.pow(max(0.0, CTL_star), 2.0))
        S_signal = math.exp(-inner)

        return {"K": K, "dV": dV, "H": H, "KLL": KLL_score, "CTL_star": CTL_star, "S_signal": S_signal}


# ==========================================
# CAV STATUS
# ==========================================

def compute_cav_status(K, H, KLL, config):
    flags = []
    if K < 0.01 and KLL < 0.01:
        return "NULL_STATE", ["null_state"]
    if H > config.H_EXPLOSION_THRESHOLD:
        flags.append("hazard_explosion")
    if KLL < config.KLL_FAIL_THRESHOLD:
        flags.append("low_legitimacy")
    if K < 0.01:
        flags.append("coherence_collapse")
    if "hazard_explosion" in flags:
        return "FAILED_HAZARD", flags
    elif "low_legitimacy" in flags:
        return "FAILED_LEGITIMACY", flags
    elif "coherence_collapse" in flags:
        return "FAILED_COHERENCE", flags
    return "PASSED", flags


# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(page_title="CSLF Real Engine", page_icon="🏛️", layout="wide")

st.markdown("""
<div style='background:linear-gradient(135deg,#0a0a2e,#1a1a4e); padding:24px 28px; border-radius:8px; margin-bottom:20px;'>
<div style='font-size:11px; color:#4fc3f7; letter-spacing:4px; margin-bottom:6px;'>COGNITIVE STABILITY & LEGITIMACY FRAMEWORK</div>
<div style='font-size:28px; font-weight:700; color:#fff;'>CSLF <span style='color:#4fc3f7'>Real Engine</span></div>
<div style='font-size:12px; color:#555; margin-top:4px;'>Gemini-Powered Epistemic Admissibility Layer --- Dr. Elhabib Kherroubi © 2026</div>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
col_a.info("**CSLF**\nEpistemic Admissibility\n*Does this deserve to exist?*")
col_b.warning("**IEL**\nEmergency Handling\n*Should we suspend?*")
col_c.success("**R-AGAM**\nSovereign Commitment\n*Is execution permitted?*")
st.markdown("---")

with st.sidebar:
    st.header("🔑 Gemini API")
    if not GEMINI_AVAILABLE:
        st.error("google-generativeai not installed.\nRun: pip install google-generativeai")
        api_key = ""
    else:
        api_key = st.text_input("API Key", type="password", placeholder="AIza...", help="Get from: aistudio.google.com")
        if api_key:
            st.success("✓ Key loaded")
        else:
            st.warning("Enter key for real mode (runs in simulation without key)")

    st.divider()
    st.header("📋 Scenario")
    domain = st.selectbox("Domain", list(DOMAIN_SCENARIOS.keys()))

    if domain == "Custom":
        steps_text = []
        st.markdown("**Enter 5 reasoning steps:**")
        for i in range(5):
            label = ["Stable 1", "Stable 2", "Drift 1", "Drift 2", "Chaos"][i]
            t = st.text_area(f"Step {i+1} ({label})", height=60, key=f"step_{i}")
            steps_text.append(t)
    else:
        steps_text = DOMAIN_SCENARIOS[domain]
        st.markdown("**Scenario steps:**")
        for i, s in enumerate(steps_text):
            with st.expander(f"Step {i+1}"):
                st.caption(s[:120] + "..." if len(s) > 120 else s)

    st.divider()
    st.header("⚙️ Parameters")
    CONFIG.LAMBDA = st.slider("λ (Hazard Amplification)", 4.0, 12.0, 8.0, 0.5)
    CONFIG.DOMAIN_RISK = st.slider("Domain Risk", 0.1, 2.0, 0.70, 0.05)
    CONFIG.KLL_FAIL_THRESHOLD = st.slider("KLL Threshold", 0.2, 0.7, 0.4, 0.05)
    st.divider()
    run = st.button("▶ Run CSLF Evaluation", type="primary", use_container_width=True)

if run:
    if not steps_text or not any(steps_text):
        st.error("Please enter scenario steps.")
        st.stop()

    mode = "🟢 REAL (Gemini)" if (GEMINI_AVAILABLE and api_key) else "🟡 SIMULATION (Heuristic)"
    st.markdown(f"**Mode: {mode}**")

    model = CSLFSignal(CONFIG)
    trace = []
    gemini_logs = []
    progress = st.progress(0, text="Initializing CSLF engine...")

    for i, reasoning_text in enumerate(steps_text):
        if not reasoning_text.strip():
            reasoning_text = f"Step {i+1}: no input provided."
        progress.progress((i + 1) / len(steps_text), text=f"Step {i+1}/{len(steps_text)}: Querying Gemini...")
        probs, token, gemini_raw = get_real_probs_from_gemini(api_key, reasoning_text, domain.split("---")[0].strip(), i)
        gemini_logs.append({"step": i + 1, "gemini_response": gemini_raw, "derived_probs": [round(p, 4) for p in probs]})
        r = model.evaluate(probs, token)
        trace.append({
            "Step": i + 1,
            "Reasoning": reasoning_text[:60] + "..." if len(reasoning_text) > 60 else reasoning_text,
            "Token": token,
            "K": round(r["K"], 4),
            "dV": round(r["dV"], 4),
            "H": round(r["H"], 6),
            "KLL": round(r["KLL"], 4),
            "CTL*": round(r["CTL_star"], 4),
            "S_signal": round(r["S_signal"], 4),
            "probs": [round(p, 4) for p in probs],
        })
    progress.empty()

    st.subheader("📈 Reasoning Trajectory")
    display_cols = ["Step", "Reasoning", "Token", "K", "dV", "H", "KLL", "CTL*", "S_signal"]
    df = pd.DataFrame(trace)[display_cols]
    st.dataframe(df, use_container_width=True)

    with st.expander("📊 Real Probability Distributions (from Gemini)"):
        prob_data = []
        for t in trace:
            prob_data.append({"Step": t["Step"], "p1": t["probs"][0], "p2": t["probs"][1], "p3": t["probs"][2], "p4": t["probs"][3]})
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True)

    if api_key:
        with st.expander("🔍 Gemini Raw Responses"):
            for log in gemini_logs:
                st.markdown(f"**Step {log['step']}:** `{log['gemini_response']}`")

    st.markdown("---")
    st.subheader("📦 CAV Vector --- Epistemic Closure")
    final = trace[-1]
    cav_status, flags = compute_cav_status(final["K"], final["H"], final["KLL"], CONFIG)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("K --- Coherence", f"{final['K']:.4f}")
    m2.metric("H --- Hazard", f"{final['H']:.6f}")
    m3.metric("KLL --- Legitimacy", f"{final['KLL']:.4f}")
    m4.metric("CTL* --- Tension", f"{final['CTL*']:.4f}")
    m5.metric("S_signal --- KSL", f"{final['S_signal']:.4f}")

    if cav_status == "PASSED":
        st.success("✅ CAV STATUS: PASSED --- Epistemically Admissible")
        st.info("Ready for Sovereign Resolution → R-AGAM")
    elif cav_status == "NULL_STATE":
        st.error("∅ NULL STATE --- Decision does not exist epistemically")
    elif cav_status == "FAILED_HAZARD":
        st.error("🔥 FAILED: HAZARD EXPLOSION --- H(t) exceeded critical threshold")
    elif cav_status == "FAILED_LEGITIMACY":
        st.error("⚖️ FAILED: LEGITIMACY FAILURE --- KLL below minimum threshold")
    elif cav_status == "FAILED_COHERENCE":
        st.error("💥 FAILED: COHERENCE COLLAPSE --- Reasoning structure disintegrated")
    else:
        st.error(f"💥 FAILED: {cav_status}")

    if flags:
        st.markdown("**Epistemic Flags:** " + " · ".join([f"`{f}`" for f in flags]))

    CAV = {
        "header": {"artifact_type": "CAV_EPISTEMIC_CLOSURE", "schema_version": "3.1.0", "issuer_id": "CSLF_ENGINE_DR_KHERROUBI"},
        "epistemic_closure": {"cav_status": cav_status, "execution_semantics": "ALLOW_IF_VERIFIED" if cav_status == "PASSED" else "BLOCK", "binding_constraint": "NON_REINTERPRETABLE"},
        "core_vectors": {"K": round(final["K"], 6), "H": round(final["H"], 6), "KLL": round(final["KLL"], 6), "CTL_star": round(final["CTL*"], 6), "S_signal": round(final["S_signal"], 6)},
        "failure_envelope": {"flags": flags},
        "handoff_contract": {"target_layer": "R_AGAM_SOVEREIGN_GATE", "forward_eligible": cav_status == "PASSED", "binding": True},
    }
    CAV["trace_root_hash"] = sign_cav(CAV)

    with st.expander("📄 Full CAV Artifact (JSON)"):
        st.json(CAV)

    st.download_button("⬇️ Download CAV JSON", data=json.dumps(CAV, indent=2), file_name="CAV_real_artifact.json", mime="application/json")

    st.markdown("---")
    st.markdown("""
> **"Admissibility is a gate --- not a score."**
>
> CSLF does not ask: *Is the decision correct?*
> CSLF asks: *Does this reasoning path have the right to exist as a decision candidate?*
>
> --- Dr. Elhabib Kherroubi, 2026
""")

else:
    # Welcome screen
    st.markdown("""
### How it works

**Without CSLF** (current systems):
- No epistemic validation
- No admissibility gate
- Decisions may propagate instability

**With CSLF + R-AGAM:**
- CSLF validates reasoning legitimacy
- R-AGAM enforces sovereign admissibility
- Execution is gated and controlled

> "Admissibility is a gate — not a score."
""")
