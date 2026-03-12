🛡️ CDEWS-IAFS
Cognitive Drift Early Warning System — Intent Anticipation & Formation System
�
�
�
�
�
�
�
Charger l'image
Charger l'image
Charger l'image
Charger l'image
Charger l'image
Charger l'image
Charger l'image
"A balance does not read intention — it reads what is written."
— Dr. Elhabib Kherroubi
📄 What is CDEWS-IAFS?
CDEWS-IAFS is an independent, deterministic monitoring layer that evaluates the structural stability of AI-generated or human-written text in real time.
While existing metrics measure fluency — CDEWS measures stability.
What current tools measure
What CDEWS measures
Fluency (BLEU, Perplexity)
Structural drift
Statistical probability
Logical collapse
Reference similarity
Structural coherence SC
⚡ Key Features
Feature
Detail
🧠 CPU Only
No GPU required — runs on any machine
🔒 No API
Fully local, zero data leakage
🌍 Multilingual
Arabic, French, English validated
📐 Deterministic
Same input = Same output, always
🔌 Model-Agnostic
Works on any LLM output
⚡ Real-Time
Instant structural analysis
🧠 Core Formula
C(t) = exp( -(α·H(t) + β·D(t) + γ·(1 - SC)) )

α = 0.45  |  β = 0.35  |  γ = 0.20
Indicator
Meaning
H(t)
Entropy — informational dispersion
D(t)
Drift — semantic deviation (Jensen-Shannon)
SC
Structural Coherence — logical consistency
🚦 Risk Classification
Zone
C(t)
Status
🟢 Green
≥ 0.82
Stable — safe for decision support
🟡 Yellow
0.48 – 0.82
Drift detected — human review needed
🔴 Red
< 0.48
Critical — do not use without review
🔗 Live Demo
👉 Try CDEWS-IAFS on Streamlit
🚀 Quick Start
git clone https://github.com/kherroubielhabib-Dr/cdews-iafs-monitor
cd cdews-iafs-monitor
pip install -r requirements.txt
streamlit run app.py
📊 Experimental Results
Text Type
C(t)
SC
Interpretation
Human — Dr. Kherroubi
0.543
1.000
Structurally coherent
Standard AI (Gemini)
0.528
0.719
Mild drift detected
Hallucinatory text
0.542
0.842
Coherent but epistemically false
Key Finding: A structurally coherent text may still be epistemically false.
Structural stability ≠ Epistemic truth.
🗺️ Roadmap
Version
Feature
Status
v3.2
Public monitoring layer
✅ Released
v6.0
Logic Coach — Structural Guidance
🔄 In development
v6.1
Document Vision Audit
🔄 In development
📖 Citation
If you use CDEWS-IAFS in your research, please cite:
@software{kherroubi2026cdews,
  author    = {Kherroubi, Elhabib},
  title     = {CDEWS-IAFS: Cognitive Drift Early Warning System},
  year      = {2026},
  version   = {3.2},
  url       = {https://github.com/kherroubielhabib-Dr/cdews-iafs-monitor},
  license   = {CC-BY-NC-4.0}
}
See also: CITATION.cff
📩 Contact
Dr. Elhabib Kherroubi — Founder & Inventor
📧 kherroubi.cdews.iafs@gmail.com
⚖️ License
This project is licensed under CC BY-NC 4.0 — free for research and non-commercial use.
For commercial licensing: kherroubi.cdews.iafs@gmail.com
CDEWS-IAFS — The balance that tells you when to trust AI.
