import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
import pandas as pd
import time
import re
from modules.nlu_pipeline import predict
from modules.data_loader import load_intents, load_eval_dataset, get_intent_names

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="BotTrainer · NLU Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080B14 !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99,102,241,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(16,185,129,0.08) 0%, transparent 55%),
        #080B14 !important;
}

[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1100px !important; }

.sidebar-brand {
    padding: 2rem 1.5rem 1.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    margin-bottom: 1rem;
}
.sidebar-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin-bottom: 0.75rem;
}
.sidebar-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.02em;
    margin: 0 0 2px !important;
}
.sidebar-sub {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #6366F1 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stRadio > label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    color: #94A3B8 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0 1.5rem;
    margin-bottom: 0.5rem !important;
}
.stRadio > div { padding: 0 0.75rem; gap: 4px !important; }
.stRadio > div > label {
    padding: 0.6rem 0.75rem !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
.stRadio > div > label:hover { background: rgba(99,102,241,0.1) !important; }

.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.page-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #F1F5F9 !important;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0 0 0.4rem !important;
}
.page-desc {
    font-size: 0.92rem !important;
    color: #64748B !important;
    margin: 0 !important;
}

.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366F1, #8B5CF6);
    border-radius: 14px 14px 0 0;
}
.metric-card:hover { border-color: rgba(99,102,241,0.3); }
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #F1F5F9;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-sub { font-size: 0.75rem; color: #475569; margin-top: 0.3rem; }

.stTextInput > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div:focus-within {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
}
.result-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
}

.intent-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 10px 18px;
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #A5B4FC;
    letter-spacing: -0.01em;
}
.intent-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #6366F1;
    box-shadow: 0 0 8px rgba(99,102,241,0.7);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

.conf-track {
    width: 100%; height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
    margin-top: 0.75rem;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #6366F1, #10B981);
}
.conf-labels {
    display: flex; justify-content: space-between;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    margin-top: 0.4rem;
}

.entity-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 0.75rem; }
.entity-tag {
    display: inline-flex; flex-direction: column;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px;
    padding: 8px 14px;
    min-width: 120px;
}
.entity-type {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #10B981;
    margin-bottom: 3px;
}
.entity-val {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    color: #6EE7B7;
}

.json-block {
    background: #0D1117;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #94A3B8;
    line-height: 1.8;
    white-space: pre;
    overflow-x: auto;
}
.json-key { color: #A5B4FC; }
.json-str { color: #6EE7B7; }
.json-num { color: #FCD34D; }

.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px;
    background: rgba(255,255,255,0.05);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

[data-testid="stVegaLiteChart"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.stSpinner > div { border-top-color: #6366F1 !important; }
hr { border-color: rgba(255,255,255,0.05) !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }

.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
}
.status-online {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.25);
    color: #10B981;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

.intent-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.intent-row:hover { border-color: rgba(99,102,241,0.25); }
.intent-row-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #A5B4FC;
    margin-bottom: 0.25rem;
}
.intent-row-desc { font-size: 0.82rem; color: #475569; margin-bottom: 0.6rem; }
.example-pill {
    display: inline-block;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.78rem;
    color: #94A3B8;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ─── Helper ─────────────────────────────────────────────────────
def highlight_json(s):
    s = re.sub(r'"(\w+)":', r'<span class="json-key">"\1"</span>:', s)
    s = re.sub(r': "([^"]+)"', r': <span class="json-str">"\1"</span>', s)
    s = re.sub(r': ([\d.]+)', r': <span class="json-num">\1</span>', s)
    return s


# ─── Load Data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    intents  = load_intents()
    eval_data = load_eval_dataset()
    return intents, eval_data

intents_data, eval_data = load_data()
intent_names = get_intent_names(intents_data)

# ─── Session State Init ─────────────────────────────────────────
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "nlu_result" not in st.session_state:
    st.session_state.nlu_result = None

# ─── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">⚡</div>
        <div class="sidebar-title">BotTrainer</div>
        <div class="sidebar-sub">NLU Studio v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["NLU Playground", "Dataset Explorer", "Evaluation"],
        label_visibility="visible"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    total_examples = sum(len(i["examples"]) for i in intents_data["intents"])
    st.markdown(f"""
    <div style="padding: 0 1.5rem;">
        <div class="metric-card" style="margin-bottom:0.75rem;">
            <div class="metric-label">Intents</div>
            <div class="metric-value" style="font-size:1.5rem;">{len(intent_names)}</div>
        </div>
        <div class="metric-card" style="margin-bottom:0.75rem;">
            <div class="metric-label">Training examples</div>
            <div class="metric-value" style="font-size:1.5rem;">{total_examples}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size:0.85rem;font-family:'DM Mono',monospace;color:#6366F1;">
                llama-3.3-70b
            </div>
            <div style="margin-top:0.5rem;">
                <span class="status-pill status-online">
                    <span class="status-dot"></span> Groq · Live
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 1 — NLU Playground
# ════════════════════════════════════════════════════════════════
if page == "NLU Playground":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">NLU Playground</div>
        <div class="page-desc">Real-time intent classification and entity extraction powered by LLaMA 3.3</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Quick Example Chips ──────────────────────────────────────
    st.markdown('<div class="section-label">Quick examples</div>', unsafe_allow_html=True)

    examples = [
        ("✈️", "Book a flight to Delhi tomorrow"),
        ("🍕", "Order me a pizza"),
        ("🌤️", "What is the weather in Chennai?"),
        ("⏰", "Set a reminder at 9pm"),
        ("🎵", "Play some jazz music"),
        ("💳", "Check my bank balance"),
        ("💬", "Send a message to John"),
        ("🗺️", "Navigate to the airport"),
    ]

    # Clicking a chip → stores text in session_state → reruns → input shows it
    cols = st.columns(4)
    for i, (icon, ex) in enumerate(examples):
        if cols[i % 4].button(f"{icon}  {ex}", key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = ex
            st.session_state.nlu_result = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Text Input ───────────────────────────────────────────────
    user_input = st.text_input(
        "User message",
        value=st.session_state.input_text,
        placeholder="Type any message, e.g. Book a flight to Mumbai next Friday…",
        key="text_input_widget"
    )
    # Keep session_state synced when user types manually
    st.session_state.input_text = user_input

    # ── Action Buttons ───────────────────────────────────────────
    col_btn, col_clear = st.columns([5, 1])
    with col_btn:
        analyze_btn = st.button("⚡  Analyze Message", type="primary", use_container_width=True)
    with col_clear:
        clear_btn = st.button("✕  Clear", use_container_width=True)

    # Clear → wipe both input text AND result, then rerun
    if clear_btn:
        st.session_state.input_text = ""
        st.session_state.nlu_result = None
        st.rerun()

    # Analyze → run pipeline, store result in session_state
    if analyze_btn:
        if st.session_state.input_text.strip():
            with st.spinner("Running NLU pipeline…"):
                start  = time.time()
                result = predict(st.session_state.input_text)
                result["_elapsed"] = round(time.time() - start, 2)
                st.session_state.nlu_result = result
        else:
            st.warning("Please enter a message or click one of the quick examples above.")

    # ── Display Results (persists across reruns via session_state) ──
    if st.session_state.nlu_result:
        result  = st.session_state.nlu_result
        elapsed = result.get("_elapsed", 0)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Analysis results</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([3, 2, 1])

        with c1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-card-title">Detected Intent</div>
                <div class="intent-badge">
                    <span class="intent-dot"></span>
                    {result['intent'].replace('_', ' ').title()}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            pct   = int(result['confidence'] * 100)
            color = "#10B981" if pct >= 90 else "#F59E0B" if pct >= 70 else "#EF4444"
            st.markdown(f"""
            <div class="result-card">
                <div class="result-card-title">Confidence Score</div>
                <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
                            color:{color};letter-spacing:-0.03em;line-height:1;">
                    {pct}<span style="font-size:1rem;color:#475569;">%</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{pct}%;"></div>
                </div>
                <div class="conf-labels"><span>0</span><span>50</span><span>100</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="result-card" style="text-align:center;">
                <div class="result-card-title">Latency</div>
                <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                            color:#F1F5F9;letter-spacing:-0.02em;line-height:1;">
                    {elapsed}s
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Entities card
        entity_count = len(result['entities'])
        st.markdown(f"""
        <div class="result-card">
            <div class="result-card-title">Extracted Entities · {entity_count} found</div>
        """, unsafe_allow_html=True)

        if result["entities"]:
            tags_html = '<div class="entity-grid">'
            for k, v in result["entities"].items():
                tags_html += f"""
                <div class="entity-tag">
                    <span class="entity-type">{k}</span>
                    <span class="entity-val">{v}</span>
                </div>"""
            tags_html += "</div>"
            st.markdown(tags_html + "</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#475569;font-size:0.85rem;padding:0.5rem 0;">
                No entities detected for this intent.
            </div></div>""", unsafe_allow_html=True)

        # Raw JSON (hide internal _elapsed key)
        st.markdown('<div class="section-label" style="margin-top:1.5rem;">Raw JSON output</div>',
                    unsafe_allow_html=True)
        display_result = {k: v for k, v in result.items() if k != "_elapsed"}
        json_str = json.dumps(display_result, indent=2)
        st.markdown(f'<div class="json-block">{highlight_json(json_str)}</div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset Explorer
# ════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Dataset Explorer</div>
        <div class="page-desc">Browse intents, training examples, and entity definitions</div>
    </div>
    """, unsafe_allow_html=True)

    total_examples = sum(len(i["examples"]) for i in intents_data["intents"])

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Total intents</div>
            <div class="metric-value">{len(intent_names)}</div>
            <div class="metric-sub">Intent classes defined</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Training examples</div>
            <div class="metric-value">{total_examples}</div>
            <div class="metric-sub">Avg {total_examples // len(intent_names)} per intent</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Entity types</div>
            <div class="metric-value">{len(intents_data['entities'])}</div>
            <div class="metric-sub">Across all intents</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Eval samples</div>
            <div class="metric-value">{len(eval_data)}</div>
            <div class="metric-sub">Labeled test cases</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Examples per intent</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame({
        "Intent":   [i["name"].replace("_", " ").title() for i in intents_data["intents"]],
        "Examples": [len(i["examples"]) for i in intents_data["intents"]]
    })
    st.bar_chart(chart_data.set_index("Intent"), color="#6366F1", height=260)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Intent browser</div>', unsafe_allow_html=True)

    selected_intent = st.selectbox(
        "Select an intent to inspect",
        intent_names,
        format_func=lambda x: x.replace("_", " ").title()
    )

    for intent in intents_data["intents"]:
        if intent["name"] == selected_intent:
            entities_str = (
                " · ".join([
                    f"<code style='background:rgba(99,102,241,0.12);color:#A5B4FC;"
                    f"padding:2px 8px;border-radius:5px;font-size:0.78rem;'>{e}</code>"
                    for e in intent["entities"]
                ])
                if intent["entities"] else
                "<span style='color:#334155;font-size:0.82rem;'>No entities</span>"
            )
            examples_html = "".join(
                [f'<span class="example-pill">{ex}</span>' for ex in intent["examples"]]
            )
            st.markdown(f"""
            <div class="intent-row">
                <div class="intent-row-name">{intent['name'].replace('_', ' ').title()}</div>
                <div class="intent-row-desc">{intent['description']}</div>
                <div style="margin-bottom:0.75rem;">{entities_str}</div>
                <div>{examples_html}</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — Evaluation
# ════════════════════════════════════════════════════════════════
elif page == "Evaluation":

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Model Evaluation</div>
        <div class="page-desc">Benchmark NLU accuracy across all 30 test samples</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-title">Evaluation config</div>
        <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            letter-spacing:0.08em;color:#475569;text-transform:uppercase;
                            margin-bottom:4px;">Model</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;color:#A5B4FC;">
                    llama-3.3-70b-versatile
                </div>
            </div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            letter-spacing:0.08em;color:#475569;text-transform:uppercase;
                            margin-bottom:4px;">Test samples</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;color:#A5B4FC;">
                    {len(eval_data)}
                </div>
            </div>
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                            letter-spacing:0.08em;color:#475569;text-transform:uppercase;
                            margin-bottom:4px;">Provider</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;color:#A5B4FC;">
                    Groq Cloud
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    run_btn = st.button("▶  Run Full Evaluation", type="primary", use_container_width=True)

    if run_btn:
        results      = []
        correct      = 0
        progress_bar = st.progress(0)
        status_text  = st.empty()

        for i, sample in enumerate(eval_data):
            status_text.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#6366F1;">'
                f'⚡ Processing {i+1}/{len(eval_data)} — '
                f'<span style="color:#94A3B8;">{sample["text"]}</span></div>',
                unsafe_allow_html=True
            )
            prediction = predict(sample["text"])
            is_correct = prediction["intent"] == sample["expected_intent"]
            if is_correct:
                correct += 1
            results.append({
                "Text":       sample["text"],
                "Expected":   sample["expected_intent"],
                "Predicted":  prediction["intent"],
                "Confidence": f"{prediction['confidence']*100:.0f}%",
                "Match":      "✅" if is_correct else "❌"
            })
            progress_bar.progress((i + 1) / len(eval_data))

        status_text.empty()
        accuracy    = correct / len(eval_data)
        score_color = "#10B981" if accuracy >= 0.9 else "#F59E0B" if accuracy >= 0.7 else "#EF4444"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value" style="color:{score_color};">{accuracy*100:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Correct</div>
                <div class="metric-value" style="color:#10B981;">{correct}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Wrong</div>
                <div class="metric-value" style="color:#EF4444;">{len(eval_data) - correct}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total tested</div>
                <div class="metric-value">{len(eval_data)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Detailed results</div>', unsafe_allow_html=True)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

        wrong_df = df[df["Match"] == "❌"]
        if not wrong_df.empty:
            st.markdown(
                '<div class="section-label" style="margin-top:1.5rem;">Incorrect predictions</div>',
                unsafe_allow_html=True
            )
            st.dataframe(wrong_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("""
            <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                        border-radius:12px;padding:1.25rem 1.5rem;text-align:center;
                        font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#10B981;">
                🎉 Perfect score — all 30 predictions correct!
            </div>
            """, unsafe_allow_html=True)