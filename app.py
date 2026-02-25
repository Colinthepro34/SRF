import streamlit as st

# 1. Page Config (Must be the first line)
st.set_page_config(page_title="NaviCore", layout="wide")

# 2. Global CSS
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

    /* Primary Buttons (Black) */
    div.stButton > button[kind="primary"] {
        background-color: black;
        color: white;
        border: none;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #222;
        border: none;
        color: white;
    }

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

    /* 3. Sticky Bottom Bar */
    .sticky-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 15px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        text-align: center;
        z-index: 999;
    }
    .sticky-btn {
        background-color: black;
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: bold;
        font-size: 16px;
        border: none;
        cursor: pointer;
        width: 100%;
        max-width: 400px;
    }
    .sticky-btn:hover {
        background-color: #333;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1.5px solid #e0e0e0;
        padding: 0.5rem 0.75rem;
        font-family: 'DM Sans', sans-serif;
    }
    .stTextInput > div > div > input:focus {
        border-color: #111;
        box-shadow: none;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("""
<nav class="ub-nav">
  <div class="ub-logo">NaviCore</div>
  <ul class="ub-nav-links">
    <li><a href="#">Main</a></li>
    <li><a href="#">Policy</a></li>
    <li><a href="#">Business</a></li>
    <li><a href="#">About</a></li>
  </ul>
  <div class="ub-nav-actions">
    <button class="ub-btn-ghost">Log in</button>
    <button class="ub-btn-solid">Sign up</button>
  </div>
</nav>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# PART 1: TOP WHITE SECTION
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("📍 **Current Location**")
    st.title("Find the shortest route now")

    start = st.text_input("Start Location", "CSMT, IN")
    end = st.text_input("Destination", placeholder="Enter destination...")

    if st.button("Calculate Route", type="primary"):
        st.success(f"Routing from {start} to {end}...")

with col2:
    st.image(
        "https://cn-geo1.uber.com/image-proc/crop/resizecrop/udam/format=auto/width=552/height=552/srcb64=aHR0cHM6Ly90Yi1zdGF0aWMudWJlci5jb20vcHJvZC91ZGFtLWFzc2V0cy80MmEyOTE0Ny1lMDQzLTQyZjktODU0NC1lY2ZmZmUwNTMyZTkucG5n",
        use_container_width=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ==========================================
# PART 2: THE BLACK SECTION
# ==========================================
st.markdown("""
<div class="black-section-container">
    <div style="display: flex; flex-wrap: wrap; gap: 2rem; align-items: center;">
        <div style="flex: 1; min-width: 300px;">
            <div class="black-section-title">PAT.ai</div>
            <div class="black-section-text">
                PAT.ai (Perform • Analyze • Transform) is a smart data assistant that lets users upload a dataset and analyze it using simple natural language commands. 
            It automatically interprets prompts to perform statistical analysis, visualization, data cleaning, and predictions..
            </div>
            <a href="#" class="white-btn">Try it</a>
        </div>
        <div style="flex: 1; min-width: 300px; text-align: center;">
            <img src="https://cn-geo1.uber.com/image-proc/crop/resizecrop/udam/format=auto/width=552/height=368/srcb64=aHR0cHM6Ly90Yi1zdGF0aWMudWJlci5jb20vcHJvZC91ZGFtLWFzc2V0cy9jNjQyNWRmNC0zMTkwLTRmZTEtODY2Ni02YTVhZjJjMGEwNDkucG5n"
                 alt="NaviCore Pro"
                 style="max-width: 100%; height: auto; border-radius: 12px;">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# PART 3: BOTTOM WHITE SECTION
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
col3, col4 = st.columns([1, 1], gap="large")

with col3:
    st.image(
        "https://cn-geo1.uber.com/image-proc/crop/resizecrop/udam/format=auto/width=552/height=311/srcb64=aHR0cHM6Ly90Yi1zdGF0aWMudWJlci5jb20vcHJvZC91ZGFtLWFzc2V0cy9kNjQ4ZjViNi1iYjVmLTQ1MGUtODczMy05MGFlZmVjYmQwOWUuanBn"
    )
with col4:
    st.title("How we works?")
    st.markdown("""
    Seamlessly add stops to pick up friends. Our algorithm recalculates the
    most efficient order to ensure everyone arrives on time without detours.
    """)
    st.page_link("pages/work.py", label="Learn more →")

st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# ==========================================
# STICKY BOTTOM BAR 
# ==========================================
st.markdown("""
    <div class="sticky-bottom">
        <button class="sticky-btn" onclick="window.scrollTo(0,0);">Calculate Route</button>
    </div>
""", unsafe_allow_html=True)