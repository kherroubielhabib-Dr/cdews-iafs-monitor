import streamlit as st
import hashlib, hmac, time, uuid, json, os, secrets, copy

# =========================================================
# 🛡️ CSLF ENGINE v5.9 — FINAL SOVEREIGN EDITION (FIXED)
# =========================================================

VERSION = "5.9"
K_MIN = 0.3
H_CRIT = 4.5
KLL_MIN = 0.4
KLL_W = 0.25
TTL_DEFAULT = 3600

# =========================
# 🔐 STATE
# =========================
if "ROOT_KEY" not in st.session_state:
    st.session_state.ROOT_KEY = os.environ.get("CSLF_ROOT_KEY") or secrets.token_hex(32)

if "NONCE_REGISTRY" not in st.session_state:
    st.session_state.NONCE_REGISTRY = {}

ROOT_KEY = st.session_state.ROOT_KEY.encode()

# =========================
# 🧠 UTILITIES
# =========================
def normalize_context(ctx):
    return json.dumps({"ctx": str(ctx).strip().lower()}, sort_keys=True)

def cleanup_nonces():
    now = time.time()
    st.session_state.NONCE_REGISTRY = {
        n: t for n, t in st.session_state.NONCE_REGISTRY.items()
        if now - t < TTL_DEFAULT
    }

# =========================
# 📚 DICTIONARIES
# =========================
HAZARD_TRIGGERS = {
    "random": 5.0, "unknown": 4.0, "unreliable": 6.0,
    "maybe": 3.5, "unstable": 5.5, "chaos": 10.0,
    "incorrect": 8.0, "guess": 7.0
}

EVIDENCE_MARKERS = {
    "shows": 1.2, "confirm": 1.5, "evidence": 1.4,
    "supported": 1.3, "consistent": 1.1,
    "verified": 1.6, "data": 1.0, "proven": 1.8
}

# =========================================================
# ⚙️ CORE ENGINE
# =========================================================
class CSLF_Engine:

    @staticmethod
    def validate_step(text):
        text_l = text.lower()
        n = max(len(text), 2)

        h_raw = sum(w for word, w in HAZARD_TRIGGERS.items() if word in text_l)
        e_score = sum(w for word, w in EVIDENCE_MARKERS.items() if word in text_l)

        h_adj = max(h_raw - (0.3 * e_score), 0)

        ratio = h_adj / n
        scaled = min(0.55 * ratio, 1.0)
        h_val = h_adj * (1 + scaled)

        k_val = 1.0 - min(h_adj / n, 1.0)

        if e_score <= 0 or k_val < 0.25:
            return {"K": round(k_val,4), "H": round(h_val,4), "KLL": 0.0}

        sa = k_val
        ci = 1.0 - min(h_val / n, 1.0)
        sv = min(e_score * k_val, 1.0)

        kll_val = (sa * ci * sv) ** KLL_W

        return {
            "K": round(k_val,4),
            "H": round(h_val,4),
            "KLL": round(kll_val,4)
        }

    @staticmethod
    def finalize_path(steps, context="NO_CONTEXT"):
        cleanup_nonces()

        results = [CSLF_Engine.validate_step(s) for s in steps]

        k_avg = round(sum(r["K"] for r in results) / len(results), 4)
        h_max = round(max(r["H"] for r in results), 4)
        kll_final = round(results[-1]["KLL"], 4)

        h_norm = min(h_max / 10.0, 1.0)
        s_signal = round((k_avg**1.2) * (1-h_norm) * (kll_final**1.1), 4)

        # Decision
        if kll_final == 0:
            status, admissible = "NULL_STATE", False
        elif h_max > H_CRIT:
            status, admissible = "FAILED_HAZARD", False
        elif k_avg < K_MIN or kll_final < KLL_MIN:
            status, admissible = "FAILED_THRESHOLD", False
        else:
            status, admissible = "PASSED", True

        trace_hash = hashlib.sha256("".join(steps).encode()).hexdigest()

        # 🔒 Freeze Hash (FIXED: same rounded values)
        freeze_payload = f"{status}-{k_avg}-{h_max}-{kll_final}"
        freeze_hash = hashlib.sha256(freeze_payload.encode()).hexdigest()

        cav = {
            "header": {
                "artifact_id": f"CAV-{uuid.uuid4().hex[:8].upper()}",
                "version": VERSION,
                "issued_at": time.time(),
                "valid_until": time.time() + TTL_DEFAULT,
                "nonce": uuid.uuid4().hex,
                "context_hash": hashlib.sha256(
                    normalize_context(context).encode()
                ).hexdigest()
            },
            "epistemic_payload": {
                "status": status,
                "admissible": admissible,
                "vectors": {
                    "K_avg": k_avg,
                    "H_max": h_max,
                    "KLL_final": kll_final,
                    "S_signal": s_signal
                },
                "freeze_hash": freeze_hash
            },
            "trace_hash": trace_hash
        }

        payload = json.dumps(cav, sort_keys=True)

        cav["cryptographic_seal"] = {
            "signature": hmac.new(ROOT_KEY, payload.encode(), hashlib.sha256).hexdigest(),
            "algo": "HMAC-SHA256"
        }

        return cav

# =========================================================
# 👑 R-AGAM HARD GATE
# =========================================================
def verify_cav(cav_obj, current_context="NO_CONTEXT"):

    if not isinstance(cav_obj, dict) or "cryptographic_seal" not in cav_obj:
        return {"verdict": "STRUCTURE_INVALID", "admissible": False}

    work = copy.deepcopy(cav_obj)
    sig = work.pop("cryptographic_seal")["signature"]

    payload = json.dumps(work, sort_keys=True)
    expected = hmac.new(ROOT_KEY, payload.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(sig, expected):
        return {"verdict": "SIG_INVALID", "admissible": False}

    ctx_hash = hashlib.sha256(normalize_context(current_context).encode()).hexdigest()
    if ctx_hash != work["header"]["context_hash"]:
        return {"verdict": "CONTEXT_DRIFT", "admissible": False}

    if work["header"]["nonce"] in st.session_state.NONCE_REGISTRY:
        return {"verdict": "REPLAY_ATTACK", "admissible": False}

    if time.time() > work["header"]["valid_until"]:
        return {"verdict": "EXPIRED", "admissible": False}

    # 🔒 Forensic Lock (FIXED)
    p = work["epistemic_payload"]

    recomputed = hashlib.sha256(
        f"{p['status']}-{p['vectors']['K_avg']}-{p['vectors']['H_max']}-{p['vectors']['KLL_final']}".encode()
    ).hexdigest()

    if recomputed != p["freeze_hash"]:
        return {"verdict": "FORENSIC_MISMATCH", "admissible": False}

    st.session_state.NONCE_REGISTRY[work["header"]["nonce"]] = time.time()

    return {
        "verdict": p["status"],
        "admissible": p["admissible"],
        "timestamp": time.time()
    }

# =========================================================
# 🎛️ UI
# =========================================================
st.set_page_config(page_title="CSLF v5.9", layout="wide")

st.title("🛡️ CSLF v5.9 — FINAL SOVEREIGN")

c1, c2 = st.columns(2)

with c1:
    path = st.text_area("Reasoning Path", height=250)
    ctx = st.text_input("Context", value="medical_protocol_2026")

    if st.button("SEAL CAV"):
        steps = [s for s in path.split("\n") if s.strip()]
        if steps:
            st.session_state.cav = CSLF_Engine.finalize_path(steps, ctx)

with c2:
    if "cav" in st.session_state:
        st.json(st.session_state.cav)

        if st.button("VERIFY"):
            result = verify_cav(st.session_state.cav, ctx)

            if result["admissible"]:
                st.success(result)
            else:
                st.error(result)

