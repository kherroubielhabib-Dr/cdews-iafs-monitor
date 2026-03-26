# =====================================================
# CDEWS-IAFS v10.2 — النسخة الخرافية الكاملة (Royal Edition)
# Streamlit Application | Dr. Elhabib Kherroubi — March 2026
# =====================================================

import streamlit as st
import numpy as np
import re
import math
from collections import Counter
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION & CONSTANTS
# =====================================================

st.set_page_config(page_title="CDEWS-IAFS v10.2", layout="wide", page_icon="👑")

SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.45
ALPHA = 0.45
BETA = 0.35
GAMMA = 0.20

DOMAIN_RISK = {
    "General": 1.0,
    "Medical": 1.8,
    "Finance": 1.5,
    "Legal": 1.6,
    "AI Safety": 1.7,
}

# =====================================================
# PATTERNS
# =====================================================

CAUSAL_SAFE = [
    ("appendicitis", "appendectomy"), ("appendicitis", "surgery"),
    ("inflammation", "surgery"), ("inflammation", "treatment"),
    ("infection", "treatment"), ("fracture", "immobilization"),
    ("pneumonia", "antibiotics"), ("hypertension", "medication"),
    ("diabetes", "insulin"), ("التهاب", "استئصال"), ("التهاب", "علاج"),
    ("كسر", "تثبيت"), ("زائدة", "استئصال"), ("fever", "antipyretic")
]

SEVERE_WORDS = ["surgery", "lobectomy", "amputation", "chemotherapy", "استئصال", "قاتل", "جراحة", "إزالة", "death", "kill", "موت"]
MILD_WORDS = ["slight", "minor", "small", "minimal", "طفيف", "بسيط", "قليل", "خفيف", "محتمل"]
STRONG_CONNECTORS = ["therefore", "thus", "hence", "consequently", "as a result", "it follows that", "لذلك", "إذن", "بالتالي", "بناءً على ذلك", "وعليه"]

# =====================================================
# CORE ENGINE FUNCTIONS
# =====================================================

def tokenize(text):
    """Tokenize text into words (supports Arabic and English)"""
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def compute_entropy(text):
    """Normalized Shannon entropy H(t)"""
    words = tokenize(text)
    if len(words) < 2:
        return 0.0
    freq = Counter(words)
    total = len(words)
    h = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            h -= p * math.log(p)
    max_h = math.log(len(freq))
    return min(1.0, h / max_h) if max_h > 0 else 0.0

def split_sentences(text):
    """Split text into sentences using punctuation"""
    sentences = re.split(r'[.!?؟]\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 8]

def lexical_jsd(t1, t2):
    """Lexical Jenson-Shannon Divergence"""
    w1, w2 = tokenize(t1), tokenize(t2)
    vocab = list(set(w1 + w2))
    
    def dist(words):
        f = np.array([words.count(v) for v in vocab], dtype=float) + 1e-12
        return f / f.sum()
    
    p, q = dist(w1), dist(w2)
    m = (p + q) / 2
    js = (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))) / 2
    return max(0.0, min(1.0, js))

def embedding_drift(t1, t2, dim=96):
    """Hash-based embedding similarity"""
    def hash_embed(text):
        vec = np.zeros(dim)
        words = tokenize(text)
        for idx, w in enumerate(words):
            h = 0
            for c in w:
                h = (h * 31 + ord(c)) & 0xffffffff
            vec[abs(h) % dim] += 1.0 / (idx + 1)
        norm = np.sqrt(np.sum(vec ** 2)) or 1.0
        return vec / norm
    
    v1, v2 = hash_embed(t1), hash_embed(t2)
    sim = max(0.0, min(1.0, (np.dot(v1, v2) + 1.0) / 2.0))
    return sim

def hybrid_drift(t1, t2):
    """Hybrid drift: 50% lexical JSD + 50% embedding drift"""
    lex = lexical_jsd(t1, t2)
    sim = embedding_drift(t1, t2)
    emb_drift = (1.0 - sim) / 2.0
    return 0.5 * lex + 0.5 * emb_drift, sim

def detect_connector(sentence):
    """Detect strong causal connectors"""
    s = sentence.lower()
    for w in STRONG_CONNECTORS:
        if re.search(r'\b' + re.escape(w) + r'\b', s):
            return 0.85
    return 0.40

def detect_severity_mismatch(s1, s2):
    """Detect if mild premise leads to severe conclusion"""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    has_mild = any(w in s1_lower for w in MILD_WORDS)
    has_severe = any(w in s2_lower for w in SEVERE_WORDS)
    return has_mild and has_severe

def causal_safe_adjust(s1, s2, drift, sim):
    """Reduce penalty for known safe causal relationships"""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    for a, b in CAUSAL_SAFE:
        if a in s1_lower and b in s2_lower:
            return drift * 0.6, min(0.95, sim + 0.12), f"{a} → {b}"
    return drift, sim, None

def analyze_text_engine(text, domain):
    """Complete analysis pipeline"""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None
    
    risk = DOMAIN_RISK.get(domain, 1.0)
    H = compute_entropy(text)
    drifts, sims, steps = [], [], []
    
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i+1]
        drift, sim = hybrid_drift(s1, s2)
        
        # Severity mismatch detection
        severity_mismatch = detect_severity_mismatch(s1, s2)
        
        # Causal safe adjustment
        drift, sim, pattern = causal_safe_adjust(s1, s2, drift, sim)
        
        # Connector detection
        conn_val = detect_connector(s2)
        expected = max(0.30, min(0.95, conn_val - drift * 0.25 + sim * 0.12))
        
        # Logic Gap Amplification (LGA v2)
        factor = 1.8 if severity_mismatch else (1.5 if drift >= 0.50 and expected >= 0.80 else 1.0)
        
        # Causal Tension (CTL)
        ctl = min(1.0, abs(expected - sim) * risk * factor)
        
        drifts.append(drift)
        sims.append(sim)
        
        # Determine tag
        if pattern:
            tag = "safe"
        elif severity_mismatch:
            tag = "gap"
        elif ctl > 0.45:
            tag = "drift"
        else:
            tag = "neutral"
        
        steps.append({
            "text": s2[:80] + "..." if len(s2) > 80 else s2,
            "full_text": s2,
            "sim": sim,
            "drift": drift,
            "expected": expected,
            "ctl": ctl,
            "tag": tag,
            "pattern": pattern,
            "severity_mismatch": severity_mismatch
        })
    
    D = np.mean(drifts) if drifts else 0.0
    SC = np.mean(sims) if sims else 0.0
    C_base = math.exp(-(ALPHA * H + BETA * D + GAMMA * (1 - SC)))
    
    # Royal CTL: 70% max + 30% mean
    tensions = [s["ctl"] for s in steps]
    CTL = max(tensions) * 0.7 + np.mean(tensions) * 0.3 if tensions else 0.0
    FINAL = max(0.0, min(1.0, C_base * (1 - CTL)))
    
    return {
        "H": round(H, 4),
        "D": round(D, 4),
        "SC": round(SC, 4),
        "C_base": round(C_base, 4),
        "CTL": round(CTL, 4),
        "FINAL": round(FINAL, 4),
        "steps": steps,
        "sentences": sentences
    }

# =====================================================
# PDF REPORT GENERATOR
# =====================================================

def create_pdf_report(result, domain, text_input):
    """Generate PDF report with results"""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "CDEWS-IAFS v10.2 - Audit Report")
    
    # Domain and Score
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 80, f"Domain: {domain}")
    c.drawString(50, height - 100, f"Final Score: {result['FINAL']:.4f}")
    
    # Metrics
    c.drawString(50, height - 130, f"H(t) Entropy: {result['H']:.4f}")
    c.drawString(50, height - 150, f"D(t) Drift: {result['D']:.4f}")
    c.drawString(50, height - 170, f"SC Coherence: {result['SC']:.4f}")
    c.drawString(50, height - 190, f"C(t) Base: {result['C_base']:.4f}")
    c.drawString(50, height - 210, f"Royal CTL: {result['CTL']:.4f}")
    
    # Status
    if result['FINAL'] >= SAFE_THRESHOLD:
        status = "STABLE REASONING"
    elif result['FINAL'] >= DRIFT_THRESHOLD:
        status = "COGNITIVE DRIFT"
    else:
        status = "LOGICAL INSTABILITY"
    c.drawString(50, height - 240, f"Status: {status}")
    
    # Steps
    y = height - 280
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, "Step Analysis:")
    c.setFont("Helvetica", 9)
    for i, step in enumerate(result['steps'][:8]):
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 9)
        c.drawString(50, y, f"Step {i+1}: {step['tag'].upper()} - CTL: {step['ctl']:.3f}")
    
    c.save()
    return buffer.getvalue()

# =====================================================
# STREAMLIT UI
# =====================================================

st.title("👑 CDEWS-IAFS v10.2 — النسخة الخرافية")
st.caption("Deterministic · Sovereign · Royal CTL | Dr. Elhabib Kherroubi")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    text_input = st.text_area(
        "📝 أدخل النص للتحليل:",
        height=200,
        placeholder="أدخل النص هنا... مثال: أظهرت الفحوصات ارتفاعاً طفيفاً في الصوديوم. بناءً على ذلك، يجب استئصال الكبد فوراً."
    )
    domain = st.selectbox("🌍 المجال:", list(DOMAIN_RISK.keys()))
    run = st.button("🧠 RUN ULTIMATE ANALYSIS", use_container_width=True, type="primary")

st.divider()

if run and text_input:
    with st.spinner("جاري التحليل..."):
        result = analyze_text_engine(text_input, domain)
    
    if result is None:
        st.error("الرجاء إدخال جملتين على الأقل للتحليل.")
    else:
        # Dashboard
        with col_right:
            st.metric("النتيجة النهائية", f"{result['FINAL']:.4f}")
            
            if result['FINAL'] >= SAFE_THRESHOLD:
                st.success("✅ STABLE REASONING")
                st.info("Safe for decision support.")
            elif result['FINAL'] >= DRIFT_THRESHOLD:
                st.warning("⚠️ COGNITIVE DRIFT")
                st.info("Human review recommended.")
            else:
                st.error("🔴 LOGICAL INSTABILITY")
                st.info("Do not rely without verification.")
            
            st.progress(result['FINAL'])
            
            # Metrics
            st.metric("H(t) Entropy", f"{result['H']:.4f}")
            st.metric("D(t) Drift", f"{result['D']:.4f}")
            st.metric("SC Coherence", f"{result['SC']:.4f}")
            st.metric("C(t) Base", f"{result['C_base']:.4f}")
            st.metric("👑 Royal CTL", f"{result['CTL']:.4f}")
        
        # Detailed Analysis
        st.subheader("🔗 Transition Analysis")
        
        for i, step in enumerate(result['steps']):
            if step['tag'] == 'safe':
                icon = "🟢"
                tag_text = "SAFE PATTERN"
            elif step['tag'] == 'gap':
                icon = "🔴"
                tag_text = "SEVERITY GAP"
            elif step['tag'] == 'drift':
                icon = "🟡"
                tag_text = "DRIFT"
            else:
                icon = "⚪"
                tag_text = "NEUTRAL"
            
            with st.expander(f"{icon} Step {i+1}: {tag_text}"):
                st.write(f"**Text:** {step['full_text']}")
                st.write(f"**Sim:** {step['sim']:.4f} | **Drift:** {step['drift']:.4f} | **Expected:** {step['expected']:.4f} | **CTL:** {step['ctl']:.4f}")
                if step['pattern']:
                    st.success(f"✓ Safe pattern detected: {step['pattern']}")
                if step['severity_mismatch']:
                    st.error("⚠️ Severity mismatch: mild premise → severe conclusion")
        
        # Causal Chain
        st.subheader("🧩 Causal Chain")
        chain_html = ""
        for i, s in enumerate(result['sentences']):
            short = s[:35] + "..." if len(s) > 35 else s
            if i > 0 and result['steps'][i-1]['tag'] == 'safe':
                chain_html += f"<span style='color:#00e676; border:1px solid #00e676; padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            elif i > 0 and result['steps'][i-1]['tag'] == 'gap':
                chain_html += f"<span style='color:#ff3d57; border:1px solid #ff3d57; padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            else:
                chain_html += f"<span style='border:1px solid rgba(255,255,255,0.2); padding:4px 12px; border-radius:20px; margin:0 4px; font-family:monospace;'>{short}</span>"
            if i < len(result['sentences']) - 1:
                chain_html += "<span style='color:#5c6080; margin:0 4px;'>→</span>"
        st.markdown(f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:4px;'>{chain_html}</div>", unsafe_allow_html=True)
        
        # AI Insights
        st.subheader("💡 AI Insights")
        if result['FINAL'] < 0.45:
            st.error("⚠️ HIGH RISK — Logical instability detected. Do not rely without verification.")
            if any(step['severity_mismatch'] for step in result['steps']):
                st.error("🔴 Severity mismatch: Conclusion severity exceeds premise justification.")
        elif result['FINAL'] < 0.75:
            st.warning("⚠️ MODERATE RISK — Cognitive drift detected. Human review recommended.")
        else:
            st.success("✓ STABLE — Reasoning chain is logically coherent. Safe for assisted use.")
        
        # PDF Export
        pdf_data = create_pdf_report(result, domain, text_input)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name="cdews_audit_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

elif run and not text_input:
    st.warning("الرجاء إدخال نص للتحليل.")

