[08:34, 11/03/2026] kherroubielhabib: # CDEWS-IAFS

Cognitive Drift Early Warning System — AI Structural Stability Monitor

## 📄 Description
CDEWS-IAFS is an independent monitoring layer that evaluates the structural stability of AI-generated or human-written text in real time. It runs locally (CPU only), requires no API, and produces deterministic, reproducible results.

## 🔗 Live Demo
https://cdews-iafs-monitor-qyfntm6j72mcwcrydrf9g6.streamlit.app

## 🧠 Core Formula
C(t) = exp(-(0.45·H(t) + 0.35·D(t) + 0.20·(1 - SC)))

## 📩 Contact
Dr. Elhabib Kherroubi — kherroubi.cdews.iafs@gmail.com
[08:47, 11/03/2026] kherroubielhabib: # 🛡️ CDEWS-IAFS
*Cognitive Drift Early Warning System — Intent Anticipation & Formation System*

> A balance does not read intention — it reads what is written.
> — Dr. Elhabib Kherroubi

---

## 📄 What is CDEWS-IAFS?

CDEWS-IAFS is an independent deterministic monitoring layer that evaluates the *structural stability* of AI-generated or human-written text in real time.

While existing metrics measure fluency — CDEWS measures stability.

---

## ⚡ Key Features

- 🧠 *CPU Only* — No GPU required
- 🔒 *No API* — Fully local, zero data leakage
- 🌍 *Multilingual* — Arabic, French, English validated
- 📐 *Deterministic* — Same input = Same output, always
- 🔌 *Model-Agnostic* — Works on any LLM output
- ⚡ *Real-Time* — Instant structural analysis

---

## 🧠 Core Formula
C(t) = exp( -(α·H(t) + β·D(t) + γ·(1 - SC)) )
α = 0.45 | β = 0.35 | γ = 0.20
| Indicator | Meaning |
|---|---|
| H(t) | Entropy — informational dispersion |
| D(t) | Drift — semantic deviation |
| SC | Structural Coherence — logical consistency |

---

## 🚦 Risk Classification

| Zone | C(t) | Status |
|---|---|---|
| 🟢 Green | ≥ 0.82 | Stable — safe for decision support |
| 🟡 Yellow | 0.48 – 0.82 | Drift detected — human review needed |
| 🔴 Red | < 0.48 | Critical — do not use without review |

---

## 🔗 Live Demo

👉 [Try CDEWS-IAFS](https://cdews-iafs-monitor-qyfntm6j72mcwcrydrf9g6.streamlit.app)

---

## 🚀 Quick Start

```bash
git clone https://github.com/kherroubielhabib-Dr/cdews-iafs-monitor
cd cdews-iafs-monitor
pip install -r requirements.txt
streamlit run app.py
📊 Experimental Results
Text Type
C(t)
SC
Human (Dr. Kherroubi)
0.543
1.000
Standard AI (Gemini)
0.528
0.719
Hallucinatory text
0.542
0.842
Key Finding: A structurally coherent text may still be epistemically false.
Structural stability ≠ Epistemic truth.
📩 Contact
Dr. Elhabib Kherroubi — Founder & Inventor
📧 kherroubi.cdews.iafs@gmail.com
🔗 LinkedIn
CDEWS-IAFS — The constitution that tells you when to trust AI.
