
  # CDEWS-IAFS v3.1 – Édition Gemini Optimisée
# Dr. Elhabib Kherroubi – kherroubi.cdews.iafs@gmail.com

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
import google.generativeai as genai
import re

SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.4

GEMINI_API_KEY = "AIzaSyCbyR_dJgjfsZRDzpWznt0m4kHl4jZdlNA"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

def normalized_entropy(probs):
    epsilon = 1e-12
    probs = np.array(probs) + epsilon
    h = -np.sum(probs * np.log(probs))
    return h / np.log(len(probs))

def js_drift(p, q):
    return jensenshannon(p, q) ** 2

def coherence_dynamic(h_norm, drift, a=0.5, b=0.5):
    return np.exp(-(a * h_norm + b * drift))

def text_to_vec(text, size=20):
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
    return (counts[:size]) / counts[:size].sum()

def analyze(text):
    C_h, H_h, D_h, resps = [], [], [], []
    prev = None
    prompts = [
        f"Analyse ce texte en 1 phrase: {text}",
        f"Evalue la fiabilite en 1 phrase: {text}",
        f"Conclusión en 1 phrase: {text}",
    ]
    for p in prompts:
        try:
            r = model.generate_content(p).text
            resps.append(r)
            vec = text_to_vec(r)
            h = normalized_entropy(vec)
            d = 0.0 if prev is None else js_drift(vec, prev)
            C_h.append(coherence_dynamic(h, d))
            H_h.append(h)
            D_h.append(d)
            prev = vec
        except Exception as e:
            C_h.append(0.5); H_h.append(0.8); D_h.append(0.1)
            resps.append(f"Erreur: {e}")
    return C_h, H_h, D_h, resps

st.set_page_config(page_title="CDEWS-IAFS", page_icon="🛡️", layout="wide")

col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.title("🛡️ CDEWS-IAFS v3.1")
    st.markdown("### Moniteur de Stabilité Cognitive — Analyse Réelle")
    st.markdown("Édition Gemini – Dr. Kherroubi")

st.divider()
st.info("🆓 Version Publique Gratuite — Version Entreprise disponible. 📩 kherroubi.cdews.iafs@gmail.com")
st.divider()

text_input = st.text_area("📝 Saisir le texte pour analyse :", height=120)
run = st.button("🚀 Lancer l'Analyse", use_container_width=True, type="primary")

if run and text_input:
    with st.spinner("🔬 Analyse Gemini en cours..."):
        C_h, H_h, D_h, resps = analyze(text_input)

    final_C = C_h[-1]
    st.subheader("📊 Tableau de Bord")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Score Final C(t)", f"{final_C:.3f}")
    k2.metric("Moyenne C(t)", f"{np.mean(C_h):.3f}")
    k3.metric("Minimum C(t)", f"{np.min(C_h):.3f}")
    k4.metric("Maximum C(t)", f"{np.max(C_h):.3f}")
    st.divider()

    st.subheader("🔎 Statut du Système")
    if final_C >= SAFE_THRESHOLD:
        st.success(f"🟢 SÉCURISÉ — C(t) = {final_C:.3f}")
    elif final_C > DRIFT_THRESHOLD:
        st.warning(f"🟡 DÉRIVE DÉTECTÉE — C(t) = {final_C:.3f}")
    else:
        st.error(f"🔴 CRITIQUE — C(t) = {final_C:.3f}")

    st.subheader("📈 Visualisation")
    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    fig.patch.set_facecolor('#0e1117')
    for ax in axes:
        ax.set_facecolor('#1a1d23')
        ax.tick_params(colors='white')
        ax.title.set_color('white')
        for s in ax.spines.values(): s.set_edgecolor('#444')
    axes[0].plot(C_h, color='#00bfff', linewidth=2)
    axes[0].axhline(y=SAFE_THRESHOLD, color='green', linestyle='--', alpha=0.7)
    axes[0].axhline(y=DRIFT_THRESHOLD, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('Cohérence C(t)'); axes[0].set_ylim(0,1)
    axes[1].plot(H_h, color='#ffa500', linewidth=2)
    axes[1].set_title('Entropie H(t)'); axes[1].set_ylim(0,1)
    axes[2].plot(D_h, color='#ff4444', linewidth=2)
    axes[2].set_title('Dérive D(t)')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🤖 Réponses Gemini")
    for i, r in enumerate(resps):
        st.markdown(f"*Étape {i+1}:* {r}")

    st.subheader("📋 Données Détaillées")
    df = pd.DataFrame({
        'Étape': range(len(C_h)),
        'Cohérence_C(t)': [round(c,4) for c in C_h],
        'Entropie_H(t)': [round(h,4) for h in H_h],
        'Dérive_D(t)': [round(d,4) for d in D_h],
    })
    st.dataframe(df)
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger CSV", csv, "cdews_rapport.csv", "text/csv")






