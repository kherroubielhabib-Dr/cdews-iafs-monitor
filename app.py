# =====================================================
# CDEWS-IAFS v3.0 – Édition Gemini RÉEL
# Moniteur de Stabilité Cognitive – Analyse Textuelle Directe
# Dr. Elhabib Kherroubi – kherroubi.cdews.iafs@gmail.com
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import google.generativeai as genai
import re

# ==============================
# Configuration
# ==============================
SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.4
STEPS = 10

# ==============================
# Clé API Gemini
# ==============================
GEMINI_API_KEY = "COLLE_TA_CLE_ICI"  # ← ضع مفتاحك هنا

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ==============================
# Moteur de Métriques
# ==============================
def normalized_entropy(probs):
    epsilon = 1e-12
    probs = np.array(probs) + epsilon
    h = -np.sum(probs * np.log(probs))
    return h / np.log(len(probs))

def js_drift(p, q):
    return jensenshannon(p, q) ** 2

def coherence_dynamic(h_norm, drift, a=0.5, b=0.5):
    instability = a * h_norm + b * drift
    return np.exp(-instability)

def text_to_probability_vector(text, size=20):
    """تحويل النص الحقيقي إلى متجه احتمالات"""
    words = re.findall(r'\w+', text.lower())
    if not words:
        return np.ones(size) / size
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: -x[1])[:size]
    counts = np.array([c for _, c in top], dtype=float)
    while len(counts) < size:
        counts = np.append(counts, 0.1)
    counts = counts[:size]
    return counts / counts.sum()

def analyze_with_gemini(text):
    """إرسال النص لـ Gemini وتحليل الاستجابات المتعددة"""
    C_history, H_history, D_history = [], [], []
    previous_vector = None
    responses = []

    prompts = [
        f"Analyse brièvement ce texte en 1 phrase: {text}",
        f"Évalue la fiabilité de cette information en 1 phrase: {text}",
        f"Donne une perspective critique en 1 phrase: {text}",
        f"Identifie les points clés en 1 phrase: {text}",
        f"Résume l'essentiel en 1 phrase: {text}",
        f"Évalue la cohérence logique en 1 phrase: {text}",
        f"Donne ton avis de confiance en 1 phrase: {text}",
        f"Identifie les incertitudes en 1 phrase: {text}",
        f"Évalue la stabilité sémantique en 1 phrase: {text}",
        f"Conclusion finale en 1 phrase: {text}",
    ]

    for prompt in prompts:
        try:
            response = model.generate_content(prompt)
            resp_text = response.text
            responses.append(resp_text)

            vec = text_to_probability_vector(resp_text)
            h = normalized_entropy(vec)

            if previous_vector is None:
                drift = 0.0
            else:
                drift = js_drift(vec, previous_vector)

            C = coherence_dynamic(h, drift)
            C_history.append(C)
            H_history.append(h)
            D_history.append(drift)
            previous_vector = vec

        except Exception as e:
            C_history.append(0.5)
            H_history.append(0.8)
            D_history.append(0.1)
            responses.append(f"Erreur: {str(e)}")

    return C_history, H_history, D_history, responses

# ==============================
# Interface
# ==============================
st.set_page_config(
    page_title="CDEWS-IAFS Monitor",
    page_icon="🛡️",
    layout="wide"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🛡️ CDEWS-IAFS v3.0")
    st.markdown("### Moniteur de Stabilité Cognitive — Analyse Réelle")
    st.markdown("Édition Gemini – Dr. Kherroubi")

st.divider()

st.info(
    "🆓 Version Publique Gratuite — "
    "Version Entreprise disponible pour déploiement institutionnel. "
    "📩 Contact : kherroubi.cdews.iafs@gmail.com"
)

st.divider()

text_input = st.text_area(
    "📝 Saisir le texte pour analyse réelle :",
    placeholder="Entrez votre texte ici — Gemini va l'analyser en temps réel...",
    height=120
)

run = st.button("🚀 Lancer l'Analyse Réelle", use_container_width=True, type="primary")

if run and text_input:
    with st.spinner("🔬 Analyse Gemini en cours — 10 requêtes réelles..."):
        C_history, H_history, D_history, responses = analyze_with_gemini(text_input)

    # KPIs
    st.subheader("📊 Tableau de Bord")
    final_C = C_history[-1]
    mean_C = np.mean(C_history)
    min_C = np.min(C_history)
    max_C = np.max(C_history)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Score Final C(t)", f"{final_C:.3f}")
    k2.metric("Moyenne C(t)", f"{mean_C:.3f}")
    k3.metric("Minimum C(t)", f"{min_C:.3f}")
    k4.metric("Maximum C(t)", f"{max_C:.3f}")

    st.divider()

    # Statut
    st.subheader("🔎 Statut du Système")
    if final_C >= SAFE_THRESHOLD:
        st.success(f"🟢 SÉCURISÉ — Score de Cohérence C(t) = {final_C:.3f}")
    elif final_C > DRIFT_THRESHOLD:
        st.warning(f"🟡 DÉRIVE DÉTECTÉE — Score de Cohérence C(t) = {final_C:.3f}")
    else:
        st.error(f"🔴 CRITIQUE : INSTABILITÉ — Score de Cohérence C(t) = {final_C:.3f}")

    # Visualisation
    st.subheader("📈 Visualisation")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0e1117')

    for ax in axes:
        ax.set_facecolor('#1a1d23')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    axes[0].plot(C_history, color='#00bfff', linewidth=2, label='C(t)')
    axes[0].axhline(y=SAFE_THRESHOLD, color='green', linestyle='--', alpha=0.7, label='Sécurisé')
    axes[0].axhline(y=DRIFT_THRESHOLD, color='red', linestyle='--', alpha=0.7, label='Critique')
    axes[0].set_title('Cohérence C(t)')
    axes[0].set_xlabel('Étape')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=7, labelcolor='white')

    axes[1].plot(H_history, color='#ffa500', linewidth=2, label='H(t)')
    axes[1].set_title('Entropie Normalisée H(t)')
    axes[1].set_xlabel('Étape')
    axes[1].set_ylim(0, 1)

    axes[2].plot(D_history, color='#ff4444', linewidth=2, label='D(t)')
    axes[2].set_title('Dérive JS D(t)')
    axes[2].set_xlabel('Étape')

    plt.tight_layout()
    st.pyplot(fig)

    # Réponses Gemini
    st.subheader("🤖 Réponses Gemini Analysées")
    for i, resp in enumerate(responses):
        st.markdown(f"*Étape {i+1}:* {resp}")

    # Données détaillées
    st.subheader("📋 Données Détaillées")
    df = pd.DataFrame({
        'Étape': range(len(C_history)),
        'Cohérence_C(t)': [round(c, 4) for c in C_history],
        'Entropie_H(t)': [round(h, 4) for h in H_history],
        'Dérive_D(t)': [round(d, 4) for d in D_history],
    })
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger le Rapport CSV", csv, "cdews_rapport.csv", "text/csv")
