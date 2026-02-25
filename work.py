import streamlit as st

# 1. Page Config (Must be the first line)
st.set_page_config(page_title="NaviCore", layout="wide")

# 2. Global CSS
# Added :root variables and removed markdown-breaking indents
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

    * { font-family: 'DM Sans', sans-serif; }

    /* Remove default top padding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 0rem !important; }

    /* Global Styles */
    body { color: black; background-color: white; }

/* ============================
       HEADER
    ============================ */
.ub-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 60px;
        background: var(--bg-primary);
        border-bottom: 1px solid var(--border);
        position: sticky;
        top: 0;
        z-index: 999;
    }
    .ub-logo {
        font-family: 'Syne', sans-serif;
        font-size: 28px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -1px;
    }
    .ub-nav-links {
        display: flex;
        gap: 36px;
        list-style: none;
        margin: 0; padding: 0;
    }
    .ub-nav-links li a {
        color: var(--text-muted);
        text-decoration: none;
        font-size: 14px;
        font-weight: 500;
        transition: color .2s;
    }
    .ub-nav-links li a:hover { color: var(--text-primary); }
    .ub-nav-actions { display: flex; gap: 16px; align-items: center; }
    .ub-btn-ghost {
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text-primary);
        padding: 9px 22px;
        border-radius: 500px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background .2s, color .2s;
    }
    .ub-btn-ghost:hover { background: var(--bg-card); }
    .ub-btn-solid {
        background: var(--btn-bg);
        border: none;
        color: var(--btn-text);
        padding: 9px 22px;
        border-radius: 500px;
        font-size: 14px;
        font-weight: 700;
        cursor: pointer;
        transition: background .2s;
    }
    .ub-btn-solid:hover { background: var(--accent-hover); }

/* MIDDLE BLACK SECTION STYLING */
.black-section-container {
    background-color: black;
    color: white;
    padding: 4rem;
    border-radius: 16px;
    margin: 2rem 0;
}
.black-section-title {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 1rem;
    letter-spacing: -0.5px;
}
.black-section-text {
    font-size: 1.1rem;
    line-height: 1.7;
    color: #aaaaaa;
    margin-bottom: 2rem;
}
.white-btn {
    background-color: white;
    color: black;
    padding: 11px 24px;
    text-decoration: none;
    font-weight: 700;
    border-radius: 8px;
    display: inline-block;
    font-size: 0.9rem;
    transition: background 0.15s;
}
.white-btn:hover {
    background-color: #f0f0f0;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<nav class="ub-nav">
  <div class="ub-logo">NaviCore</div>
</nav>
""", 
unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# PART 2: THE BLACK SECTION
# ==========================================
# Fixed the broken image tag and removed indents
st.markdown("""
<div class="black-section-container">
<div style="display: flex; flex-wrap: wrap; gap: 2rem; align-items: center;">
<div style="flex: 1; min-width: 300px;">
<div class="black-section-title">Working</div>
<div class="black-section-text">
PAT.ai (Perform • Analyze • Transform) is a smart data assistant that lets users upload a dataset and analyze it using simple natural language commands.
It automatically interprets prompts to perform statistical analysis, visualization, data cleaning, and predictions.
</div>
</div>
""", unsafe_allow_html=True)