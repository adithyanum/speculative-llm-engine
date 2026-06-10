import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

load_dotenv("SpecDec.env")

# ── Page config ───────────────────────────────────────────────────────────────
# Must be the FIRST streamlit call — before any st.anything
st.set_page_config(
    page_title="SpecDecode AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────────────────────────────
# We inject raw CSS via st.markdown with unsafe_allow_html=True.
# Streamlit renders your Python top-to-bottom on every interaction,
# so this CSS block runs first and styles everything below it.
#
# Key decisions:
#   --bg        : deep navy-black, darker than Neural Edge's #080b10
#   --accent    : #00d4aa teal — one hot colour, used sparingly
#   --blue      : #5b9eff for latency metrics only
#   --amber     : #e8a820 for cycle counts only
#   JetBrains Mono throughout — same as Neural Edge, it's correct for this
#   Syne for the logo only — display weight, not body text
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');

:root {
    --bg:         #07090f;
    --bg2:        #040810;
    --bg3:        #060810;
    --border:     #0d1520;
    --border2:    #0a1422;
    --text:       #c8d4ec;
    --text-dim:   #1e3a50;
    --accent:     #00d4aa;
    --accent-dim: #001a14;
    --accent-b:   #003828;
    --blue:       #5b9eff;
    --blue-dim:   #020c1a;
    --amber:      #e8a820;
    --red:        #ef5050;
    --mono:       'JetBrains Mono', monospace;
    --display:    'Syne', sans-serif;
}

* { box-sizing: border-box; }

.stApp {
    background-color: var(--bg);
    color: var(--text);
    font-family: var(--mono);
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 1rem; max-width: 1280px; }

[data-testid="stSidebar"] {
    background-color: #060810;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--mono);
    color: var(--text);
}

.sb-label {
    font-size: 9px;
    color: #152030;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: var(--mono);
}

.logo-wrap {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.logo-name {
    font-family: var(--display);
    font-size: 1.6rem;
    font-weight: 800;
    color: #eef4ff;
    letter-spacing: -0.03em;
    line-height: 1;
}
.logo-name span { color: var(--accent); }
.logo-sub {
    font-size: 0.68rem;
    color: #132030;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 5px;
    font-family: var(--mono);
}

div[data-testid="stTabs"] button {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: #1e3a50 !important;
    letter-spacing: 0.04em;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stTextArea textarea {
    background: var(--bg2) !important;
    color: #7aacbf !important;
    border: 0.5px solid #0d1c2a !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 13px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px #00d4aa20 !important;
}
.stTextArea textarea::placeholder { color: #0e2030 !important; }

.stTextInput input {
    background: var(--bg2) !important;
    color: #7aacbf !important;
    border: 0.5px solid #0d1c2a !important;
    border-radius: 7px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #001a14 !important;
    border: none !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stRadio"] label {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: #2a4a60 !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    color: var(--accent) !important;
}

[data-testid="stSlider"] label,
[data-testid="stSlider"] div {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    color: #1e3a4a !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }

[data-testid="stMetric"] {
    background: var(--bg2) !important;
    border: 0.5px solid var(--border2) !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 9px !important;
    color: #132830 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 15px !important;
    color: var(--accent) !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 10px !important;
}

.user-msg {
    display: inline-block;
    background: #04111e;
    border: 0.5px solid #0a1e30;
    border-radius: 10px 10px 2px 10px;
    padding: 9px 14px;
    font-size: 12px;
    color: #6a9abf;
    max-width: 68%;
    font-family: var(--mono);
    margin-bottom: 4px;
}

.resp-head {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.rh-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
}
.rh-tag {
    font-size: 10px;
    color: var(--accent);
    letter-spacing: 0.07em;
    font-family: var(--mono);
}
.rh-meta {
    font-size: 10px;
    color: #133828;
    font-family: var(--mono);
}

.resp-box {
    background: var(--bg2);
    border: 0.5px solid var(--border2);
    border-left: 2px solid #00d4aa40;
    border-radius: 0 8px 8px 8px;
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 13px;
    color: #7aacbf;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
    position: relative;
    overflow: hidden;
}
.resp-box::after {
    content: '';
    position: absolute;
    top: -40%; left: 0; right: 0;
    height: 40%;
    background: linear-gradient(to bottom, transparent, rgba(0,212,170,0.03), transparent);
    animation: scan 1.6s ease-out both;
    pointer-events: none;
}
@keyframes scan {
    0%   { top: -40%; }
    100% { top: 110%; }
}

.resp-box-base {
    background: var(--bg2);
    border: 0.5px solid var(--border2);
    border-left: 2px solid #5b9eff40;
    border-radius: 0 8px 8px 8px;
    padding: 14px 16px;
    font-family: var(--mono);
    font-size: 13px;
    color: #6a9abf;
    line-height: 1.75;
    white-space: pre-wrap;
    word-break: break-word;
}
.rh-dot-base {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--blue);
    flex-shrink: 0;
}
.rh-tag-base {
    font-size: 10px;
    color: var(--blue);
    letter-spacing: 0.07em;
    font-family: var(--mono);
}

.compare-bar {
    background: #030c0a;
    border: 0.5px solid #0a2018;
    border-radius: 8px;
    padding: 11px 15px;
    display: flex;
    gap: 0;
    margin-top: 14px;
    animation: fadeUp 0.4s ease both;
}
.cb-cell {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0 14px;
    border-right: 1px solid #0d1a14;
    font-family: var(--mono);
}
.cb-cell:first-child { padding-left: 0; }
.cb-cell:last-child  { border-right: none; }
.cb-l  { font-size: 9px; color: #133028; text-transform: uppercase; letter-spacing: 0.07em; }
.cb-pos { font-size: 13px; font-weight: 500; color: var(--accent); }
.cb-neg { font-size: 13px; font-weight: 500; color: var(--red); }
.cb-neu { font-size: 13px; font-weight: 500; color: var(--blue); }
.cb-dim { font-size: 11px; color: #1e4038; }

@keyframes fadeUp   { from { opacity:0; transform:translateY(6px); }  to { opacity:1; transform:translateY(0); } }
@keyframes fadeDown { from { opacity:0; transform:translateY(-6px); } to { opacity:1; transform:translateY(0); } }

.stAlert { font-family: var(--mono) !important; font-size: 12px !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_BACKEND = os.getenv("BACKEND", "")

DOMAIN_PREFIXES = {
    "General":         "",
    "Exam · PSC/UPSC": "Answer briefly and accurately as if for a PSC/UPSC exam: ",
    "Tech · DSA/CS":   "Explain clearly for a computer science student: ",
    "Language":        "Translate the following to Malayalam: ",
}

# ── API helpers ───────────────────────────────────────────────────────────────

def get_url():
    return st.session_state.get("backend_url", DEFAULT_BACKEND)

def check_health(url):
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def call_generate(url, prompt, mode, max_tokens):
    # timeout=120 — T4 generation can legitimately take 60-90s
    try:
        r = requests.post(
            f"{url}/generate",
            json={"prompt": prompt, "mode": mode, "max_new_tokens": max_tokens},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": r.json().get("detail", f"HTTP {r.status_code}")}
    except requests.exceptions.Timeout:
        return {"error": "Timed out — backend may still be loading models."}
    except Exception as e:
        return {"error": str(e)}



# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.2rem 0 1rem">
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                        color:#eef4ff;letter-spacing:-0.02em;line-height:1">
                Spec<span style="color:#00d4aa">Decode</span>
            </div>
            <div style="font-size:9px;color:#132030;letter-spacing:0.07em;
                        text-transform:uppercase;margin-top:5px;font-family:'JetBrains Mono',monospace">
                inference engine · v2.0
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # Backend URL — persisted in session_state across reruns
        st.markdown('<div class="sb-label">Backend URL</div>', unsafe_allow_html=True)
        url = st.text_input(
            label="url",
            value=st.session_state.get("backend_url", DEFAULT_BACKEND),
            label_visibility="collapsed",
        )
        st.session_state["backend_url"] = url

        health = check_health(url)
        if health and health.get("models_loaded"):
            st.success(f"● connected · {health.get('device', '?')}")
        elif health:
            st.warning("● server up · models not loaded")
        else:
            st.error("● unreachable")

        st.divider()

        # Mode — format_func maps clean API string → display label
        st.markdown('<div class="sb-label">Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            label="mode",
            options=["speculative", "baseline", "compare"],
            format_func=lambda x: {
                "speculative": "⚡  Speculative",
                "baseline":    "▤  Baseline (7B only)",
                "compare":     "⊞  Compare both",
            }[x],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown('<div class="sb-label">Domain</div>', unsafe_allow_html=True)
        domain = st.radio(
            label="domain",
            options=list(DOMAIN_PREFIXES.keys()),
            label_visibility="collapsed",
        )

        st.divider()

        # step=50 keeps values clean — no 137 or 263 floating around
        st.markdown('<div class="sb-label">Max tokens</div>', unsafe_allow_html=True)
        max_tokens = st.slider(
            label="tokens",
            min_value=50, max_value=500,
            value=200, step=50,
            label_visibility="collapsed",
        )

    return mode, domain, max_tokens

# ── Response renderers ────────────────────────────────────────────────────────

def render_response_block(result, mode):
    """
    Single-mode renderer.
    resp-box gets the teal left border + scan animation.
    resp-box-base gets the blue left border, no animation — visually quieter.
    Metrics differ: speculative shows acceptance + cycles, baseline doesn't have them.
    """
    if mode == "speculative":
        ar     = result.get("acceptance_rate") or 0
        cycles = result.get("cycles") or "—"
        st.markdown(f"""
        <div class="resp-head">
            <div class="rh-dot"></div>
            <span class="rh-tag">SPECULATIVE · 0.5B + 7B</span>
            <span class="rh-meta">· {cycles} cycles · {ar*100:.1f}% accepted</span>
        </div>
        <div class="resp-box">{result['response']}</div>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TOK / SEC", f"{result['tokens_per_sec']:.2f}")
        c2.metric("LATENCY",   f"{result['latency']:.1f}s")
        c3.metric("ACCEPT",    f"{ar*100:.1f}%")
        c4.metric("CYCLES",    str(cycles))
    else:
        st.markdown(f"""
        <div class="resp-head">
            <div class="rh-dot-base"></div>
            <span class="rh-tag-base">BASELINE · 7B ONLY</span>
        </div>
        <div class="resp-box-base">{result['response']}</div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("TOK / SEC", f"{result['tokens_per_sec']:.2f}")
        c2.metric("LATENCY",   f"{result['latency']:.1f}s")
        c3.metric("TOKENS",    str(result['tokens_generated']))


def render_compare_block(spec, base):
    """
    Compare mode — side by side with the delta bar below.
    ThreadPoolExecutor already ran before this is called;
    we just render the two results here.
    The delta bar auto-explains when baseline wins (short prompt note).
    """
    col1, col2 = st.columns(2, gap="large")
    spec_tps  = spec.get("tokens_per_sec", 0)
    base_tps  = base.get("tokens_per_sec", 0)
    spec_wins = spec_tps > base_tps

    with col1:
        ar     = spec.get("acceptance_rate") or 0
        cycles = spec.get("cycles") or "—"
        trophy = " · winner" if spec_wins else ""
        st.markdown(f"""
        <div class="resp-head">
            <div class="rh-dot"></div>
            <span class="rh-tag">SPECULATIVE · 0.5B + 7B{trophy}</span>
            <span class="rh-meta">· {cycles} cycles</span>
        </div>
        <div class="resp-box">{spec['response']}</div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("TOK / SEC", f"{spec_tps:.2f}", delta=f"{spec_tps - base_tps:+.2f}")
        c2.metric("ACCEPT",    f"{ar*100:.1f}%")
        c3.metric("CYCLES",    str(cycles))

    with col2:
        trophy = " · winner" if not spec_wins else ""
        st.markdown(f"""
        <div class="resp-head">
            <div class="rh-dot-base"></div>
            <span class="rh-tag-base">BASELINE · 7B ONLY{trophy}</span>
        </div>
        <div class="resp-box-base">{base['response']}</div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("TOK / SEC", f"{base_tps:.2f}", delta=f"{base_tps - spec_tps:+.2f}")
        c2.metric("LATENCY",   f"{base.get('latency', 0):.1f}s")
        c3.metric("TOKENS",    str(base.get("tokens_generated", "—")))

    # Delta bar
    tps_d    = spec_tps - base_tps
    lat_d    = (spec.get("latency", 0)) - (base.get("latency", 0))
    winner   = "speculative" if spec_wins else "baseline"
    tps_cls  = "cb-pos" if tps_d >= 0 else "cb-neg"
    lat_cls  = "cb-neg" if lat_d > 0 else "cb-pos"
    tps_sign = "+" if tps_d >= 0 else ""
    lat_sign = "+" if lat_d >= 0 else ""

    st.markdown(f"""
    <div class="compare-bar">
        <div class="cb-cell">
            <div class="cb-l">Δ throughput</div>
            <div class="{tps_cls}">{tps_sign}{tps_d:.2f} tok/s</div>
        </div>
        <div class="cb-cell">
            <div class="cb-l">Δ latency</div>
            <div class="{lat_cls}">{lat_sign}{lat_d:.2f}s</div>
        </div>
        <div class="cb-cell">
            <div class="cb-l">winner</div>
            <div class="cb-neu">{winner}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Generate tab ──────────────────────────────────────────────────────────────
def render_generate_tab(mode, domain, max_tokens):
    url = get_url()

    # Config strip — active settings visible above the prompt box
    st.markdown(f"""
    <div style="display:flex;gap:6px;margin-bottom:12px;font-family:'JetBrains Mono',monospace">
        <span style="font-size:10px;padding:3px 9px;border-radius:999px;
                     color:#00d4aa;border:0.5px solid #00d4aa30;background:#001a14">
            ⚡ {mode}
        </span>
        <span style="font-size:10px;padding:3px 9px;border-radius:999px;
                     color:#1e3a50;border:0.5px solid #0a1422;background:transparent">
            {domain}
        </span>
        <span style="font-size:10px;padding:3px 9px;border-radius:999px;
                     color:#5b9eff;border:0.5px solid #5b9eff30;background:#020c1a">
            {max_tokens} tok
        </span>
    </div>
    """, unsafe_allow_html=True)

    prompt_raw = st.text_area(
        label="prompt",
        placeholder="Ask something...   (try: explain binary search · translate hello to Malayalam)",
        height=100,
        label_visibility="collapsed",
    )

    # User bubble — renders immediately after typing, before hitting generate
    if prompt_raw.strip():
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin:6px 0 2px">'
            f'<div class="user-msg">{prompt_raw.strip()}</div></div>',
            unsafe_allow_html=True,
        )

    col_btn, col_tip = st.columns([1, 4])
    with col_btn:
        run = st.button("generate ⚡", use_container_width=True, type="primary")
    with col_tip:
        if mode == "compare":
            st.caption("fires speculative + baseline in parallel · ThreadPoolExecutor")
        elif mode == "speculative":
            st.caption("draft: Qwen2.5-0.5B  ·  target: Qwen2.5-7B  ·  k=4")
        else:
            st.caption("target only: Qwen2.5-7B  ·  standard autoregressive")

    if run and not prompt_raw.strip():
        st.warning("enter a prompt first.")
        return

    if run:
        full_prompt = DOMAIN_PREFIXES[domain] + prompt_raw.strip()
        st.divider()

        if mode == "compare":
            # ThreadPoolExecutor fires both API calls simultaneously.
            # .submit() is non-blocking → returns a Future.
            # .result() blocks until the thread finishes.
            # Total wait = max(spec_time, base_time), not their sum.
            with st.spinner("running both modes in parallel..."):
                with ThreadPoolExecutor(max_workers=2) as ex:
                    sf = ex.submit(call_generate, url, full_prompt, "speculative", max_tokens)
                    bf = ex.submit(call_generate, url, full_prompt, "baseline",    max_tokens)
                    spec = sf.result()
                    base = bf.result()
            if "error" in spec:
                st.error(f"speculative error: {spec['error']}")
                return
            if "error" in base:
                st.error(f"baseline error: {base['error']}")
                return
            render_compare_block(spec, base)

        else:
            spinner_msg = (
                "drafting with 0.5B, verifying with 7B..."
                if mode == "speculative"
                else "generating with 7B baseline..."
            )
            with st.spinner(spinner_msg):
                result = call_generate(url, full_prompt, mode, max_tokens)
            if "error" in result:
                st.error(f"error: {result['error']}")
                return
            render_response_block(result, mode)


# ── Main ──────────────────────────────────────────────────────────────────────
# No tabs — the generate view is the whole app.
# render_sidebar returns mode/domain/max_tokens which flow directly in.
def main():
    mode, domain, max_tokens = render_sidebar()

    st.markdown("""
    <div class="logo-wrap">
        <div class="logo-name">
            Spec<span>Decode</span>
            <span style="color:#5b9eff;font-size:1rem;font-weight:500"> AI</span>
        </div>
        <div class="logo-sub">Speculative decoding inference engine · Qwen2.5 0.5B draft + 7B target</div>
    </div>
    """, unsafe_allow_html=True)

    render_generate_tab(mode, domain, max_tokens)


if __name__ == "__main__":
    main()