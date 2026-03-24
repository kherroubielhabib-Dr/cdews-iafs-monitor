# =====================================================
# CDEWS-IAFS v5.4 — Reasoning Integrity Engine
# CTL v4.3 backbone + LNI diagnostic + DRA + Hybrid D(t)
# Dr. Elhabib Kherroubi — March 2026
# =====================================================
#
# WHAT'S NEW IN v5.4:
# ✅ Code Mode — detects programming language input
#    → disables LNI / connector detection (not applicable)
#    → disables length-jump penalty (normal in code)
#    → evaluates only SC + C(t) + structural CTL
#    → adds code-specific flags: orphan blocks, dead branches
# ✅ applymap → map (pandas deprecation fix)
# ✅ Missing "no contradictions" feedback added
# ✅ Missing "at least 2 sentences" message restored
#
# FULL LINEAGE:
# v3.2  → H(t) · D(t) · SC · C(t)
# v4.1  → CTL — Causal Tension Layer
# v4.2  → DRA — Domain Risk Amplification
# v4.3  → Deterministic Core
# v4.3.1→ Formula precision
# v4.4  → Semantic D(t) + domain-aware H(t)
# v5.0  → DCS + LNI (architecture — DCS symmetric issue)
# v5.1  → CTL v4.3 backbone restored + LNI as diagnostic
# v5.2  → Hybrid D(t): lexical JSD + embedding cosine
# v5.3  → Demo Mode decision layer + high-risk domain gate
# v5.4  → Code Mode + pandas fix + UI consistency fixes
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIG
# =====================================================
SAFE_THRESHOLD  = 0.75
DRIFT_THRESHOLD = 0.45

ALPHA = 0.45   # Entropy weight
BETA  = 0.35   # Drift weight
GAMMA = 0.20   # Structural Coherence weight

DOMAIN_RISK = {
    "General":   1.0,
    "Medical":   1.8,
    "Finance":   1.5,
    "Legal":     1.6,
    "AI Safety": 1.7,
}

# Connector strength — calibrated against PDTB (Prasad et al. 2008)
CONNECTOR_STRENGTH = {
    "strong":   0.85,
    "medium":   0.65,
    "contrast": 0.50,
    "neutral":  0.40,
}

CAUSAL_CONNECTORS = {
    "strong": [
        "therefore", "thus", "hence", "consequently", "as a result",
        "it follows that", "accordingly", "for this reason",
        "لذلك", "إذن", "بالتالي", "بناءً على ذلك", "وعليه", "من ثم",
        "نتيجة لذلك", "على هذا الأساس", "يترتب على ذلك",
        "donc", "par conséquent", "ainsi", "en conséquence",
    ],
    "medium": [
        "because", "since", "given that", "due to",
        "لأن", "بسبب", "نظراً لـ", "بما أن",
        "parce que", "car", "puisque",
    ],
    "contrast": [
        "but", "however", "yet", "although", "nevertheless", "despite",
        "لكن", "غير أن", "مع ذلك", "رغم ذلك", "بيد أن",
        "mais", "cependant", "pourtant", "néanmoins",
    ],
}

# =====================================================
# CODE MODE — detection patterns
# =====================================================
CODE_PATTERNS = [
    r'\bdef\s+\w+\s*\(',          # Python function
    r'\bclass\s+\w+',             # Python/JS class
    r'\bimport\s+\w+',            # import statement
    r'\bif\s+\w+.*:',             # Python if block
    r'\breturn\s+',               # return statement
    r'^\s*(const|let|var)\s+',    # JS variables
    r'#\s*={3,}',                 # comment dividers
    r'st\.\w+\(',                 # Streamlit calls
    r'np\.\w+\(',                 # NumPy calls
    r'pd\.\w+\(',                 # Pandas calls
]

def detect_code_mode(text: str) -> bool:
    """
    Detects if the input is programming code rather than natural language.
    Returns True if ≥ 3 code patterns are found.
    Threshold = 3 to avoid false positives on technical prose.
    """
    matches = sum(1 for p in CODE_PATTERNS if re.search(p, text, re.MULTILINE))
    return matches >= 3

# =====================================================
# MODEL
# =====================================================
@st.cache_resource
def load_model():
    with st.spinner("🧠 Initializing CDEWS v5.4 Reasoning Engine..."):
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# =====================================================
# UTILITIES
# =====================================================
def split_sentences(text: str) -> list:
    parts = re.split(r'[.!?؟\n]+', text)
    return [p.strip() for p in parts if len(p.strip()) > 10]

def batch_embed(sentences: list) -> list:
    vecs = model.encode(sentences, convert_to_numpy=True).astype(np.float32)
    return [vecs[i] for i in range(len(vecs))]

# =====================================================
# CORE METRICS
# =====================================================
def compute_entropy(text: str, is_technical: bool = False) -> float:
    """
    H(t) — Normalized Shannon entropy.
    Domain-aware: technical texts × 0.70 correction.
    """
    words = re.findall(r'\w+', text.lower())
    if len(set(words)) < 2:
        return 0.0
    vocab = sorted(set(words))
    freq  = np.array([words.count(w) for w in vocab], dtype=float)
    freq += 1e-12
    freq /= freq.sum()
    raw = -np.sum(freq * np.log(freq))
    h = float(np.clip(raw / np.log(len(freq)), 0.0, 1.0))
    return h * 0.70 if is_technical else h

def lexical_drift(text1: str, text2: str) -> float:
    """
    Lexical drift via word-frequency JSD.
    """
    from scipy.spatial.distance import jensenshannon
    words1 = set(re.findall(r"\w+", text1.lower()))
    words2 = set(re.findall(r"\w+", text2.lower()))
    vocab  = sorted(words1 | words2)

    def to_aligned(text):
        words = re.findall(r"\w+", text.lower())
        freq  = np.array([words.count(w) for w in vocab], dtype=float)
        freq += 1e-12
        return freq / freq.sum()

    p = to_aligned(text1)
    q = to_aligned(text2)
    return float(np.clip(jensenshannon(p, q) ** 2, 0.0, 1.0))

def semantic_drift_embedding(e1: np.ndarray, e2: np.ndarray) -> float:
    sim = float(cosine_similarity([e1], [e2])[0][0])
    return float(np.clip((1.0 - sim) / 2.0, 0.0, 1.0))

def compute_semantic_drift(
    e1: np.ndarray, e2: np.ndarray,
    text1: str = "", text2: str = ""
) -> float:
    """
    D(t) v5.2 — Hybrid: 0.5×lexical_JSD + 0.5×embedding_cosine_distance
    """
    emb_drift = semantic_drift_embedding(e1, e2)
    if text1 and text2:
        lex = lexical_drift(text1, text2)
        return float(np.clip(0.5 * lex + 0.5 * emb_drift, 0.0, 1.0))
    return emb_drift

def compute_structural_coherence(embeddings: list) -> float:
    """SC — Mean cosine similarity across consecutive pairs."""
    if len(embeddings) < 2:
        return 1.0
    sims = []
    for i in range(len(embeddings) - 1):
        sim = float(cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0])
        sims.append(float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0)))
    return float(np.mean(sims))

def compute_c_score(h: float, d: float, sc: float) -> float:
    """C(t) = exp(-(α·H + β·D + γ·(1-SC))). Deterministic."""
    return float(np.exp(-(ALPHA * h + BETA * d + GAMMA * (1.0 - sc))))

# =====================================================
# v5.1 COMPONENTS
# =====================================================
def detect_connector(sentence: str):
    s = sentence.lower()
    for ctype, words in CAUSAL_CONNECTORS.items():
        for w in words:
            if w in s:
                return CONNECTOR_STRENGTH[ctype], ctype, w
    return CONNECTOR_STRENGTH["neutral"], "neutral", "---"

def compute_lni(expected: float, drift: float) -> float:
    """
    LNI — Logical Necessity Index (DIAGNOSTIC ONLY in v5.1+)
    LNI = expected × (1 − drift)
    NOT used in CTL calculation. Shown as diagnostic only.
    """
    return float(np.clip(expected * (1.0 - drift), 0.0, 1.0))

# =====================================================
# CTL ENGINE — v4.3 backbone (standard mode)
# =====================================================
def compute_ctl(
    sentences: list,
    embeddings: list,
    domain: str,
    code_mode: bool = False
) -> tuple:
    """
    CTL v5.4 — Causal Tension Layer.

    Standard mode (text):
        base    = |expected − actual_sim|
        CTL     = base × risk(domain)
        penalty = sim × 0.75 if length_ratio > 1.5

    Code mode:
        - No connector detection (code has no causal connectors)
        - No length-jump penalty (long functions are normal)
        - CTL = embedding drift × risk(domain)
        - LNI suppressed (not meaningful for code)
        - Adds code-specific flags: orphan block, dead branch
    """
    if len(sentences) < 2:
        return 0.0, []

    risk     = DOMAIN_RISK.get(domain, 1.0)
    tensions = []
    details  = []

    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i + 1]
        e1, e2 = embeddings[i], embeddings[i + 1]

        drift = compute_semantic_drift(e1, e2, s1, s2)

        if code_mode:
            # ── CODE MODE CTL ──────────────────────────────────
            # Pure embedding drift — no connector expectation
            ctl = float(np.clip(drift * risk, 0.0, 1.0))
            lni = None   # Not applicable

            # Code-specific flags
            has_return = bool(re.search(r'\breturn\b', s1))
            unreachable = has_return and len(s2.strip()) > 10
            orphan = bool(re.search(r'^\s*(else|elif|except|finally)\b', s2))

            if ctl > 0.6:
                flag = "💣 Semantic Gap"
            elif unreachable:
                flag = "⚠️ Possible Dead Branch"
            elif orphan:
                flag = "🔥 Orphan Block"
            elif ctl > 0.2:
                flag = "⚠️ Medium Tension"
            else:
                flag = "✅ Stable"

            tensions.append(ctl)
            details.append({
                "Step":      i + 1,
                "Connector": "N/A (Code)",
                "Type":      "code",
                "Expected":  "N/A",
                "Sim":       round(1.0 - drift, 3),
                "Drift":     round(drift, 3),
                "LNI":       "N/A",
                "CTL":       round(ctl, 3),
                "Flag":      flag,
            })

        else:
            # ── STANDARD MODE CTL (v4.3 backbone) ─────────────
            expected, ctype, connector = detect_connector(s2)

            sim = float(np.clip(
                (cosine_similarity([e1], [e2])[0][0] + 1.0) / 2.0,
                0.0, 1.0
            ))

            # Jump penalty (v4.1+) — disabled in code mode
            if len(s2) / max(len(s1), 1) > 1.5:
                sim *= 0.75

            base = abs(expected - sim)
            ctl  = float(np.clip(base * risk, 0.0, 1.0))
            lni  = compute_lni(expected, drift)

            contradiction = (sim < 0.25 and expected > 0.75)

            tensions.append(ctl)
            details.append({
                "Step":      i + 1,
                "Connector": connector,
                "Type":      ctype,
                "Expected":  round(expected, 3),
                "Sim":       round(sim, 3),
                "Drift":     round(drift, 3),
                "LNI":       round(lni, 3),
                "CTL":       round(ctl, 3),
                "Flag": (
                    "🔴 CONTRADICTION" if contradiction else
                    "💣 Semantic Gap"  if ctl > 0.5   else
                    "🔥 High"          if ctl > 0.35  else
                    "⚠️ Medium Tension" if ctl > 0.2  else
                    "✅ Stable"
                ),
            })

    return float(np.mean(tensions)) if tensions else 0.0, details

# =====================================================
# FULL ANALYSIS PIPELINE
# =====================================================
def analyze(
    text: str,
    domain: str,
    is_technical: bool = False,
    code_mode: bool = False
) -> dict:
    sentences  = split_sentences(text)
    if not sentences:
        return {}

    embeddings = batch_embed(sentences)
    h  = compute_entropy(text, is_technical or code_mode)
    d  = float(np.mean([
        compute_semantic_drift(embeddings[i], embeddings[i+1],
                               sentences[i], sentences[i+1])
        for i in range(len(embeddings) - 1)
    ])) if len(embeddings) >= 2 else 0.0

    sc     = compute_structural_coherence(embeddings)
    c_base = compute_c_score(h, d, sc)
    ctl, details = compute_ctl(sentences, embeddings, domain, code_mode)
    final  = float(np.clip(c_base * (1.0 - ctl), 0.0, 1.0))

    return {
        "H":         round(h, 4),
        "D":         round(d, 4),
        "SC":        round(sc, 4),
        "C_base":    round(c_base, 4),
        "CTL":       round(ctl, 4),
        "Final":     round(final, 4),
        "details":   details,
        "code_mode": code_mode,
    }

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="CDEWS-IAFS v5.4", layout="wide")
st.title("🧠 CDEWS-IAFS v5.4 — Reasoning Integrity Engine")
st.caption(
    "Dr. Elhabib Kherroubi | "
    "Deterministic · Model-Agnostic · Sovereign | "
    "CTL v4.3 + LNI Diagnostic + DRA + Hybrid D(t) + Code Mode"
)
st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text = st.text_area(
        "📝 Enter text for analysis:",
        height=220,
        placeholder="Enter any text in Arabic, French, or English — or paste code...",
    )

with col_right:
    domain       = st.selectbox("🌍 Domain:", list(DOMAIN_RISK.keys()))
    is_technical = st.checkbox(
        "📚 Technical / Expert text",
        help=(
            "Enable for scientific papers, medical reports, legal documents. "
            "Applies 0.70 entropy correction to avoid penalizing "
            "specialized vocabulary."
        )
    )
    auto_code = st.checkbox(
        "🖥️ Auto-detect Code Mode",
        value=True,
        help="Automatically switches to Code Mode when programming code is detected."
    )
    st.info(f"Risk multiplier: **×{DOMAIN_RISK[domain]}**")

analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
st.divider()

if analyze_btn:
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Auto-detect code mode
        code_mode = auto_code and detect_code_mode(text)

        if code_mode:
            st.info("🖥️ **Code Mode activated** — connector detection and LNI suppressed. Evaluating structural coherence only.")

        with st.spinner("Computing reasoning integrity..."):
            result = analyze(text, domain, is_technical, code_mode)

        if not result:
            st.error("Could not parse text. Please enter at least one sentence.")
        else:
            final = result["Final"]

            # =========================
            # 🧠 DECISION LAYER
            # =========================
            st.markdown("## 🧠 Reasoning Decision")

            if domain in ["Medical", "AI Safety"]:
                st.warning("⚠️ High-risk domain: stricter validation applied")

            if final >= SAFE_THRESHOLD:
                st.success("✅ Reasoning is Stable")
                st.markdown("""
This output follows a coherent logical structure
and is safe for assisted use.

✔ No critical reasoning gaps detected
✔ Logical transitions are consistent
                """)
            elif final >= DRIFT_THRESHOLD:
                st.warning("⚠️ Early Signs of Instability")
                st.markdown("""
The reasoning appears mostly valid,
but contains weak or unclear transitions.

→ Human review is recommended
                """)
            else:
                st.error("🚫 Hidden Logical Risk Detected")
                st.markdown("""
This output contains a reasoning gap
that may lead to incorrect conclusions.

→ Do NOT rely on this output without verification
                """)

            if domain in ["Medical", "AI Safety"] and final < SAFE_THRESHOLD:
                st.error("🔒 Output Restricted due to high-risk domain")

            st.divider()

            # =========================
            # 🔍 TECHNICAL DETAILS
            # =========================
            with st.expander("🔍 View Technical Details"):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("H(t)", result["H"])
                c2.metric("D(t)", result["D"])
                c3.metric("SC",   result["SC"])
                c4.metric("C_base", result["C_base"])
                c5.metric("CTL",  result["CTL"])

                st.markdown(f"### 🚀 Final Score: **{final}**")

                if result["details"]:
                    df = pd.DataFrame(result["details"])

                    def color_flags(val):
                        if "💣" in str(val):
                            return 'background-color: #ff6b6b; color: white'
                        elif "🔴" in str(val):
                            return 'background-color: #ff4444; color: white'
                        elif "🔥" in str(val):
                            return 'background-color: #ffa500'
                        elif "⚠️" in str(val):
                            return 'background-color: #ffd700'
                        return ''

                    # ✅ FIXED: applymap → map (pandas deprecation fix)
                    styled_df = df.style.map(color_flags, subset=['Flag'])
                    st.dataframe(styled_df, use_container_width=True)

                    # Summary
                    logic_gaps    = [d for d in result["details"] if "💣" in d.get("Flag", "")]
                    contradictions= [d for d in result["details"] if "CONTRADICTION" in d.get("Flag", "")]
                    dead_branches = [d for d in result["details"] if "Dead Branch" in d.get("Flag", "")]
                    low_lni       = [d for d in result["details"]
                                     if isinstance(d.get("LNI"), float) and d["LNI"] < 0.3]

                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        if logic_gaps:
                            st.error(f"💣 {len(logic_gaps)} Logic Gap(s)")
                        else:
                            st.success("✅ No logic gaps")

                    with col_b:
                        if contradictions:
                            st.error(f"🔴 {len(contradictions)} Contradiction(s)")
                        elif dead_branches:
                            st.warning(f"⚠️ {len(dead_branches)} Possible Dead Branch(es)")
                        else:
                            # ✅ FIXED: was missing in v5.3
                            st.success("✅ No contradictions")

                    with col_c:
                        if low_lni:
                            st.warning(f"⚠️ {len(low_lni)} low LNI steps")
                        elif not code_mode:
                            st.success("✅ Logical necessity OK")
                        else:
                            st.info("ℹ️ LNI not applicable in Code Mode")

                    # Export
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Export Results (CSV)",
                        data=csv,
                        file_name="cdews_analysis_results.csv",
                        mime="text/csv",
                    )
                else:
                    # ✅ FIXED: was missing in v5.3
                    st.info("At least 2 sentences required for CTL analysis.")

# =====================================================
# FORMULA REFERENCE
# =====================================================
with st.expander("📐 Formula Reference & Scientific Justification"):
    st.markdown("""
**Core Stability:**
```
C(t) = exp(-(α·H(t) + β·D(t) + γ·(1 - SC)))
α = 0.45 | β = 0.35 | γ = 0.20
```

**Hybrid D(t) v5.2:**
```
D(t) = 0.5 × Lexical_JSD + 0.5 × Embedding_cosine_distance
```

**Causal Tension — Standard Mode (CTL v4.3):**
```
Base = |Expected − Actual_sim|
CTL  = Base × Risk(domain)
```

**Causal Tension — Code Mode (v5.4):**
```
CTL  = Embedding_drift × Risk(domain)
(No connector expectation, no jump penalty)
```

**Logical Necessity Index (LNI — diagnostic, text mode only):**
```
LNI = expected × (1 − drift)
```
*LNI is shown in the step table as a diagnostic signal.*
*It explains WHY tension is high — it does not modify CTL.*
*Not applicable in Code Mode.*

**Final Score:**
```
Final = C(t) × (1 − CTL)
```

---
**Connector Strength** *(Penn Discourse Treebank, Prasad et al. 2008)*

| Type | Value | Linguistic Basis |
|------|-------|-----------------|
| Strong (therefore, thus...) | 0.85 | Entailment — conclusion MUST follow |
| Medium (because, since...)  | 0.65 | Causation — conclusion LIKELY follows |
| Contrast (but, however...)  | 0.50 | Opposition — semantic distance expected |
| Neutral (no connector)      | 0.40 | Weak expectation only |

---
**Version Lineage:**

| Version | Key Addition |
|---------|-------------|
| v3.2    | H(t) · D(t) · SC · C(t) |
| v4.1    | CTL — Causal Tension Layer |
| v4.2    | DRA — Domain Risk Amplification |
| v4.3    | Deterministic Core |
| v4.4    | Semantic D(t) + domain-aware H(t) |
| v5.0    | DCS + LNI architecture |
| v5.1    | CTL v4.3 backbone restored + LNI as diagnostic |
| v5.2    | Hybrid D(t): lexical JSD + embedding cosine |
| v5.3    | Demo Mode decision layer + high-risk domain gate |
| v5.4    | Code Mode + pandas fix + UI consistency ✅ |

---
*Domain Risk: General ×1.0 | Finance ×1.5 | Legal ×1.6 | AI Safety ×1.7 | Medical ×1.8*
    """)
