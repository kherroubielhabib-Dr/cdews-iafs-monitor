# =====================================================
# CDEWS-IAFS v3.2 – Autonomous Core + Gemini Hybrid
# Moniteur de Stabilité Cognitive
# Dr. Elhabib Kherroubi – kherroubi.cdews.iafs@gmail.com
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import re
import math
from collections import Counter

# ==============================
# Configuration
# ==============================
SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.4
GEMINI_API_KEY = "COLLE_TA_CLE_ICI"  # ← ضع مفتاحك هنا (اختياري)

# ==============================
# MOTEUR LOCAL AUTONOME
# ==============================

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def char_entropy(text):
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
    return entropy / max_entropy if max_entropy > 0 else 0.0

def word_entropy(tokens):
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    max_entropy = math.log2(len(freq)) if len(freq) > 1 else 1
    return entropy / max_entropy if max_entropy > 0 else 0.0

def lexical_drift(tokens):
    if len(tokens) < 4:
        return 0.0
    mid = len(tokens) // 2
    first_half = Counter(tokens[:mid])
    second_half = Counter(tokens[mid:])
    vocab = list(set(list(first_half.keys()) + list(second_half.keys())))
    if not vocab:
        return 0.0
    p = np.array([first_half.get(w, 0) for w in vocab], dtype=float)
    q = np.array([second_half.get(w, 0) for w in vocab], dtype=float)
    p = p / p.sum() if p.sum() > 0 else np.ones(len(vocab)) / len(vocab)
    q = q / q.sum() if q.sum() > 0 else np.ones(len(vocab)) / len(vocab)
    return float(jensenshannon(p, q) ** 2)

def sentence_coherence(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 2:
        return 1.0
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    cv = std_len / mean_len if mean_len > 0 else 0
    return max(0.0, 1.0 - min(cv, 1.0))

def structural_complexity(tokens):
    if not tokens:
        return 0.0
    avg_len = np.mean([len(t) for t in tokens])
    return min(avg_len / 10.0, 1.0)

def local_analyze(text):
    tokens = tokenize(text)
    if not tokens:
        return [], [], [], ["Texte vide"]

    H_char = char_entropy(text)
    H_word = word_entropy(tokens)
    H = (H_char + H_word) / 2

    D = lexical_drift(tokens)
    SC = sentence_coherence(text)
    Cx = structural_complexity(tokens)

    instability = 0.4 * H + 0.3 * D + 0.15 * (1 - SC) + 0.15 * Cx
    C_final = math.exp(-instability)

    steps = min(len(tokens), 10)
    C_history, H_history, D_history = [], [], []
    prev_vec = None

    chunk_size = max(1, len(tokens) // steps)
    for i in range(steps):
        chunk = tokens[i * chunk_size: (i + 1) * chunk_size]
        if not chunk:
            chunk = tokens[-1:]

        vocab = list(set(tokens))
        freq = Counter(chunk)
        vec = np.array([freq.get(w, 0) for w in vocab], dtype=float)
        vec = vec / vec.sum() if vec.sum() > 0 else np.ones(len(vocab)) / len(vocab)

        h_step = word_entropy(chunk)

        if prev_vec is None or len(prev_vec) != len(vec):
            d_step = 0.0
        else:
            d_step = float(jensenshannon(vec, prev_vec) ** 2)

        c_step = math.exp(-(0.5 * h_step + 0.5 * d_step))
        C_history.append(c_step)
        H_history.append(h_step)
        D_history.append(d_step)
        prev_vec = vec

    C_history[-1] = C_final

    analysis_text = [
        f"Entropie caractères: {H_char:.3f} | Entropie mots: {H_word:.3f}",
        f"Dérive lexicale JS: {D:.3f} | Cohérence structurelle: {SC:.3f}",
        f"Complexité: {Cx:.3f} | Score Final C(t): {C_final:.3f}",
    ]

    return C_history, H_history, D_history, analysis_text

# ==============================
# MOTEUR GEMINI (Optionnel)
# ==============================

def gemini_analyze(text, api_key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash-lite")

        C_h, H_h, D_h, resps = [], [], [], []
        prev = None
        prompts = [
            f"Analyse ce texte en 1 phrase: {text}",
            f"Evalue la fiabilite en 1 phrase: {text}",
            f"Conclusión en 1 phrase: {text}",
        ]

        vocab = tokenize(text)
        if not vocab:
            vocab = ["default"]

        for p in prompts:
            r = model.generate_content(p).text
            resps.append(r)
            tokens_r = tokenize(r)
            freq = Counter(tokens_r)
            vec = np.array([freq.get(w, 0) for w in vocab], dtype=float)
            vec = (vec + 0.01) / (vec + 0.01).sum()

            h = word_entropy(tokens_r)
            d = 0.0 if prev is None else float(jensenshannon(vec, prev) ** 2)
            C_h.append(math.exp(-(0.5 * h + 0.5 * d)))
            H_h.append(h)
            D_h.append(d)
            prev = vec

        return C_h, H_h, D_h, resps, True

    except Exception as e:
        return None, None, None, [str(e)], False

# ==============================
# INTERFACE STREAMLIT
# ==============================

st.set_page_config(page_title="CDEWS-IAFS", page_icon="🛡️", layout="wide")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🛡️ CDEWS-IAFS v3.2")
    st.markdown("### Moniteur de Stabilité Cognitive")
    st.markdown("*Autonomous Core + Gemini Hybrid* — Dr. Kherroubi")

st.divider()
st.info("🆓 Version Publique Gratuite — Version Entreprise disponible. 📩 kherroubi.cdews.iafs@gmail.com")
st.divider()

with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    engine = st.radio("🔧 Moteur d'Analyse", ["🧠 Local Autonome", "🌐 Gemini (API)"])
    if engine == "🌐 Gemini (API)":
        user_key = st.text_input("Clé API Gemini", type="password")
    st.markdown("---")
    st.markdown("*Seuils de Décision*")
    safe_th = st.number_input("Seuil Sécurisé", 0.0, 1.0, SAFE_THRESHOLD, 0.05)
    drift_th = st.number_input("Seuil Critique", 0.0, 1.0, DRIFT_THRESHOLD, 0.05)
    st.markdown("---")
    st.info("v3.2 — Autonomous Core\nFonctionne sans Internet")

text_input = st.text_area(
    "📝 Saisir le texte pour analyse :",
    placeholder="Entrez votre texte ici — le système analyse localement sans API...",
    height=140
)

run = st.button("🚀 Lancer l'Analyse", use_container_width=True, type="primary")

if run and text_input:

    gemini_used = False

    if engine == "🌐 Gemini (API)" and user_key:
        with st.spinner("🌐 Tentative Gemini..."):
            C_h, H_h, D_h, resps, success = gemini_analyze(text_input, user_key)
        if success:
            gemini_used = True
            st.success("✅ Analyse Gemini réussie")
        else:
            st.warning(f"⚠️ Gemini indisponible — Basculement vers moteur local. ({resps[0][:80]})")

    if not gemini_used:
        with st.spinner("🧠 Analyse locale en cours..."):
            C_h, H_h, D_h, resps = local_analyze(text_input)
        st.info("🧠 Moteur Local Autonome — Aucune API requise")

    final_C = C_h[-1]

    st.subheader("📊 Tableau de Bord")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Score Final C(t)", f"{final_C:.3f}")
    k2.metric("Moyenne C(t)", f"{np.mean(C_h):.3f}")
    k3.metric("Minimum C(t)", f"{np.min(C_h):.3f}")
    k4.metric("Maximum C(t)", f"{np.max(C_h):.3f}")
    st.divider()

    st.subheader("🔎 Statut du Système")
    if final_C >= safe_th:
        st.success(f"🟢 SÉCURISÉ — Score de Cohérence C(t) = {final_C:.3f}")
    elif final_C > drift_th:
        st.warning(f"🟡 DÉRIVE DÉTECTÉE — Score de Cohérence C(t) = {final_C:.3f}")
    else:
        st.error(f"🔴 CRITIQUE : INSTABILITÉ — Score de Cohérence C(t) = {final_C:.3f}")

    st.subheader("📈 Visualisation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#1a1d23')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        for s in ax.spines.values():
            s.set_edgecolor('#444')

    axes[0].plot(C_h, color='#00bfff', linewidth=2.5, marker='o', markersize=4)
    axes[0].axhline(y=safe_th, color='green', linestyle='--', alpha=0.7, label='Sécurisé')
    axes[0].axhline(y=drift_th, color='red', linestyle='--', alpha=0.7, label='Critique')
    axes[0].set_title('Cohérence C(t)')
    axes[0].set_xlabel('Étape')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, labelcolor='white', facecolor='#1a1d23')

    axes[1].plot(H_h, color='#ffa500', linewidth=2.5, marker='o', markersize=4)
    axes[1].set_title('Entropie H(t)')
    axes[1].set_xlabel('Étape')
    axes[1].set_ylim(0, 1)

    axes[2].plot(D_h, color='#ff4444', linewidth=2.5, marker='o', markersize=4)
    axes[2].set_title('Dérive JS D(t)')
    axes[2].set_xlabel('Étape')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🔬 Analyse Détaillée")
    for i, r in enumerate(resps):
        st.markdown(f"*{i+1}.* {r}")

    st.subheader("📋 Données Numériques")
    df = pd.DataFrame({
        'Étape': range(len(C_h)),
        'Cohérence_C(t)': [round(c, 4) for c in C_h],
        'Entropie_H(t)': [round(h, 4) for h in H_h],
        'Dérive_D(t)': [round(d, 4) for d in D_h],
    })
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger le Rapport CSV", csv, "cdews_rapport_v32.csv", "text/csv")







