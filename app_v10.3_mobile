# =====================================================
# CDEWS-IAFS v10.3 Mobile FINAL
# Dr. Elhabib Kherroubi — March 2026
# =====================================================

import numpy as np
import re
import math
from collections import Counter

# =====================================================
# CONFIG
# =====================================================

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
# MEDICAL LAYER (MRL v2.1 FINAL)
# =====================================================

SYMPTOMS = ["ألم","حمى","حرارة","قيء","غثيان","صداع","تعب","ضعف","pain","fever","nausea"]
CONDITIONS = ["التهاب","زائدة","عدوى","كسر","سرطان","appendicitis","inflammation","infection","fracture"]
PROCEDURES = ["استئصال","جراحة","عملية","تثبيت","علاج","appendectomy","surgery","operation","treatment"]

def get_medical_confidence(text):
    t = text.lower()
    
    has_s = any(w in t for w in SYMPTOMS)
    has_c = any(w in t for w in CONDITIONS)
    has_p = any(w in t for w in PROCEDURES)
    
    score = (0.3 if has_s else 0) + (0.4 if has_c else 0) + (0.3 if has_p else 0)
    
    if has_s and has_c and has_p:
        status = "VALID_FLOW"
    elif has_s and has_p and not has_c:
        status = "MISSING_DIAGNOSIS"
    else:
        status = "UNCERTAIN"
        
    return score, status

# =====================================================
# CORE FUNCTIONS
# =====================================================

def tokenize(text):
    return re.findall(r'[\u0600-\u06FF\w]+', text.lower())

def split_sentences(text):
    text = re.sub(r'(Dr|Mr|Mrs|Prof)\.', r'\1PROTECT', text)
    text = re.sub(r'(\d+)\.(\d+)', r'\1DECIMAL\2', text)
    
    sentences = re.split(r'[.!?؟]\s+', text)
    sentences = [s.replace('PROTECT','.').replace('DECIMAL','.') for s in sentences]
    
    return [s.strip() for s in sentences if len(s.strip()) > 6]

def compute_entropy(text):
    words = tokenize(text)
    if len(words) < 2:
        return 0.0
    
    freq = Counter(words)
    total = len(words)
    
    h = 0
    for c in freq.values():
        p = c / total
        h -= p * math.log(p)
    
    max_h = math.log(len(freq))
    H = min(1.0, h / max_h) if max_h > 0 else 0
    
    # Mobile correction
    if len(words) < 10:
        H *= 0.55
    elif len(words) < 20:
        H *= 0.75
        
    return H

def hybrid_drift(s1, s2):
    w1, w2 = set(tokenize(s1)), set(tokenize(s2))
    
    if not w1 or not w2:
        return 0.5, 0.5
    
    sim = len(w1 & w2) / len(w1 | w2)
    drift = 1 - sim
    
    # length correction
    l1, l2 = len(w1), len(w2)
    ratio = min(l1, l2) / max(l1, l2)
    if ratio < 0.5:
        drift = min(1.0, drift * 1.2)
    
    return drift, sim

# =====================================================
# EXPECTATION ENGINE (NEW)
# =====================================================

def estimate_expected(s):
    s = s.lower()
    
    if any(w in s for w in ["لذلك","بالتالي","وعليه","thus","therefore"]):
        return 0.85
    elif any(w in s for w in ["قد","محتمل","ربما","maybe","possible"]):
        return 0.45
    else:
        return 0.60

# =====================================================
# MAIN ENGINE
# =====================================================

def analyze_mobile(text, domain="General"):
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return None
    
    risk = DOMAIN_RISK.get(domain, 1.0)
    H = compute_entropy(text)
    
    drifts, sims, tensions = [], [], []
    accumulated = ""
    
    for i in range(len(sentences)-1):
        s1, s2 = sentences[i], sentences[i+1]
        accumulated += " " + s1
        
        drift, sim = hybrid_drift(s1, s2)
        expected = estimate_expected(s2)
        
        ctl = min(1.0, abs(expected - sim) * risk)
        
        # --- MRL intervention (context-aware) ---
        if domain == "Medical":
            m_conf, m_status = get_medical_confidence(accumulated + " " + s2)
            
            if m_status == "VALID_FLOW":
                ctl *= 0.45
                H *= 0.7
                drift *= 0.75
                
            elif m_status == "MISSING_DIAGNOSIS":
                ctl *= 1.25
        
        drifts.append(drift)
        sims.append(sim)
        tensions.append(ctl)
    
    D = np.mean(drifts)
    SC = np.mean(sims)
    
    C_base = math.exp(-(ALPHA*H + BETA*D + GAMMA*(1-SC)))
    
    # Royal CTL p90
    CTL = np.percentile(tensions, 90) if tensions else 0
    
    # Stability Layer (improved)
    if len(drifts) > 1:
        jumps = [abs(drifts[i]-drifts[i-1]) for i in range(1,len(drifts))]
        stability = (np.std(drifts) + np.mean(jumps)) * 0.15
    else:
        stability = 0
    
    FINAL = C_base * (1 - CTL) - stability
    
    # --- MRL integration ---
    if domain == "Medical":
        m_conf, m_status = get_medical_confidence(text)
        
        if m_status == "VALID_FLOW":
            FINAL = FINAL * 0.75 + m_conf * 0.25
        elif m_status == "MISSING_DIAGNOSIS":
            FINAL *= 0.85
    
    FINAL = max(0, min(1, FINAL))
    
    # Status
    if FINAL >= SAFE_THRESHOLD:
        status = "STABLE"
    elif FINAL >= DRIFT_THRESHOLD:
        status = "DRIFT"
    else:
        status = "CRITICAL"
    
    return {
        "score": round(FINAL,4),
        "status": status,
        "entropy": round(H,4),
        "drift": round(D,4),
        "coherence": round(SC,4),
        "ctl": round(CTL,4),
        "stability": round(stability,4)
    }

# =====================================================
# QUICK TEST
# =====================================================

if __name__ == "__main__":
    text = "ارتفاع الحرارة + ألم في الربع السفلي الأيمن. التهاب الزائدة. نوصي باستئصال فوري."
    
    res = analyze_mobile(text, "Medical")
    
    print("CDEWS v10.3 Mobile FINAL")
    print(res)

