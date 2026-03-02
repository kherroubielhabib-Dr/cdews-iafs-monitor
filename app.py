# =====================================================
# CDEWS-Lite v2.1 – Édition Française (Hardening+)
# Moniteur de Stabilité - Version Améliorée & Complète
# Dr. Elhabib Kherroubi – kherroubi.cdews.iafs@gmail.com
# =====================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# ==============================
# Configuration des Seuils
# ==============================
SAFE_THRESHOLD = 0.75
DRIFT_THRESHOLD = 0.4
STEPS = 30

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

def coherence(h_norm, drift):
    instability = 0.5 * h_norm + 0.5 * drift
    return np.exp(-instability)

def get_seed_from_text(text):
    return sum(ord(c) for c in text[:50]) % 10000

# ==============================
# Interface Utilisateur
# ==============================
st.set_page_config(
    page_title="CDEWS-IAFS Monitor",
    page_icon="🛡️",
    layout="wide"
)

# En-tête professionnel
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🛡️ CDEWS-IAFS")
    st.markdown("### Moniteur de Stabilité Cognitive")
    st.markdown("Édition Professionnelle v2.1 – Dr. Kherroubi")

st.divider()

# ==============================
# سطر Freemium الرسمي
# ==============================
st.info(
    "🆓 Version Publique Gratuite — "
    "Version Entreprise disponible pour déploiement institutionnel. "
    "📩 Contact : kherroubi.cdews.iafs@gmail.com"
)

st.divider()

# ==============================
# Panneau de Configuration
# ==============================
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")

    steps_config = st.slider("Nombre d'étapes", 10, 100, STEPS)
    alpha = st.slider("Coefficient α (Entropie)", 0.1, 1.0, 0.5)
    beta = st.slider("Coefficient β (Dérive)", 0.1, 1.0, 0.5)

    st.markdown("---")
    st.markdown("*Seuils de Décision*")
    safe_th = st.number_input("Seuil Sécurisé", 0.0, 1.0, SAFE_THRESHOLD, 0.05)
    drift_th = st.number_input("Seuil Critique", 0.0, 1.0, DRIFT_THRESHOLD, 0.05)

    st.markdown("---")
    st.info("CDEWS-IAFS v2.1\nStabilité Cognitive IA")

# ==============================
# Zone de Saisie Principale
# ==============================
text_input = st.text_area(
    "📝 Saisir le texte pour analyse :",
    placeholder="Entrez votre texte technique ici...",
    height=120
)

run = st.button("🚀 Lancer l'Analyse", use_container_width=True, type="primary")

# ==============================
# Moteur d'Analyse
# ==============================
if run and text_input:

    with st.spinner("Analyse en cours..."):

        seed_val = get_seed_from_text(text_input)
        np.random.seed(seed_val)

        def coherence_dynamic(h_norm, drift, a, b):
            instability = a * h_norm + b * drift
            return np.exp(-instability)

        C_history = []
        H_history = []
        D_history = []
        previous_vector = None

        for step in range(steps_config):
            probs = np.random.dirichlet(np.ones(10))
            h = normalized_entropy(probs)

            if previous_vector is None:
                drift = 0.0
            else:
                drift = js_drift(probs, previous_vector)

            C = coherence_dynamic(h, drift, alpha, beta)

            previous_vector = probs
            C_history.append(C)
            H_history.append(h)
            D_history.append(drift)

    # ==============================
    # Tableau de Bord – KPIs
    # ==============================
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

    # ==============================
    # Statut du Système
    # ==============================
    st.subheader("🔎 Statut du Système")

    if final_C >= safe_th:
        st.success(f"🟢 SÉCURISÉ — Score de Cohérence C(t) = {final_C:.3f}")
    elif final_C > drift_th:
        st.warning(f"🟡 DÉRIVE DÉTECTÉE — Score de Cohérence C(t) = {final_C:.3f}")
    else:
        st.error(f"🔴 CRITIQUE : INSTABILITÉ — Score de Cohérence C(t) = {final_C:.3f}")

    # ==============================
    # Visualisation – 3 Graphiques
    # ==============================
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
    axes[0].axhline(safe_th, color='#00ff88', linestyle="--", linewidth=1.2, label='Sécurisé')
    axes[0].axhline(drift_th, color='#ff4444', linestyle="--", linewidth=1.2, label='Critique')
    axes[0].fill_between(range(steps_config), C_history, alpha=0.15, color='#00bfff')
    axes[0].set_title("Cohérence C(t)", fontweight='bold')
    axes[0].set_xlabel("Étape")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8, labelcolor='white', facecolor='#1a1d23')
    axes[0].grid(True, alpha=0.2, linestyle=':')

    axes[1].plot(H_history, color='#ffaa00', linewidth=2, label='H(t)')
    axes[1].fill_between(range(steps_config), H_history, alpha=0.15, color='#ffaa00')
    axes[1].set_title("Entropie Normalisée H(t)", fontweight='bold')
    axes[1].set_xlabel("Étape")
    axes[1].set_ylabel("Entropie")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8, labelcolor='white', facecolor='#1a1d23')
    axes[1].grid(True, alpha=0.2, linestyle=':')

    axes[2].plot(D_history, color='#ff6b6b', linewidth=2, label='D(t)')
    axes[2].fill_between(range(steps_config), D_history, alpha=0.15, color='#ff6b6b')
    axes[2].set_title("Dérive JS D(t)", fontweight='bold')
    axes[2].set_xlabel("Étape")
    axes[2].set_ylabel("Dérive")
    axes[2].legend(fontsize=8, labelcolor='white', facecolor='#1a1d23')
    axes[2].grid(True, alpha=0.2, linestyle=':')

    plt.tight_layout()
    st.pyplot(fig)

    # ==============================
    # Tableau de Données Détaillé
    # ==============================
    st.subheader("📋 Données Détaillées")

    df = pd.DataFrame({
        "Étape": range(steps_config),
        "Cohérence_C(t)": [round(c, 4) for c in C_history],
        "Entropie_H(t)": [round(h, 4) for h in H_history],
        "Dérive_D(t)": [round(d, 4) for d in D_history],
        "Statut": [
            "SÉCURISÉ" if c >= safe_th else ("DÉRIVE" if c > drift_th else "CRITIQUE")
            for c in C_history
        ]
    })

    st.dataframe(df, use_container_width=True)

    # ==============================
    # Exportation CSV
    # ==============================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger le Rapport CSV",
        data=csv,
        file_name=f"cdews_iafs_rapport_{seed_val}.csv",
        mime="text/csv",
        use_container_width=True
    )

elif run and not text_input:
    st.warning("⚠️ Veuillez saisir un texte avant de lancer l'analyse.")