import streamlit as st
import pandas as pd
from collections import Counter
import altair as alt


# Page configuration
st.set_page_config(
    page_title="Job Market Dashboard",
    layout="wide"
)

st.markdown("""
<div class="header-container">
    <div class="header-icon">💼</div>
    <div class="header-title">Job Market Dashboard</div>
    <div class="header-subtitle">Malaysia Job Data Analytics</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Import custom font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
    /* Apply the font globally */
    html, body, [class*="css"], div, p, span, label, input, textarea, button, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
    }
    

    /* Global styles */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .header-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        font-weight: 400;
    }
    
    /* .metric-box {
        padding: 12px;
        border-radius: 8px;
        border: 2px solid #4facfe;
        background-color: rgba(79,172,254,0.08);
        text-align: left;
    }
    */    

    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        font-size: 0.95rem;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_df(path):
    return pd.read_csv(path)

# --- Load data ---
df = load_df("https://huggingface.co/datasets/seemarn/jobstreet/resolve/main/jobstreet_all_job_dataset_2025_skills_ner_clustered.csv")

# --- Identify skills column ---
skills_col = [col for col in df.columns if "ner_skill" in col.lower()][0]

# Convert stringified lists to actual lists
df[skills_col] = df[skills_col].apply(
    lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
)

# --- Compute top 10 skills ---
all_skills = [s for sub in df[skills_col] if isinstance(sub, list) for s in sub]
skill_counts = Counter(all_skills).most_common(10)
skills, counts = zip(*skill_counts)
df_skill_counts = pd.DataFrame({"Skill": skills, "Count": counts})

# --- Session state ---
if "selected_skill" not in st.session_state:
    st.session_state.selected_skill = None

# --- PAGE 1: Overview ---
if st.session_state.selected_skill is None:
    st.title("📊 Top 10 High-Demand Skills")
    st.caption("Click a skill below to explore detailed job insights.")

    # Altair bar chart
    bars = (
        alt.Chart(df_skill_counts)
        .mark_bar()
        .encode(
            x=alt.X("Skill:N", sort="-y", title="Skill"),
            y=alt.Y("Count:Q", title="Job Postings"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Skill", "Count"]
        )
        .properties(width=900, height=500)
    )
    st.altair_chart(bars, use_container_width=True)

    # Skill selection buttons
    st.subheader("🔍 Explore Skills")
    cols = st.columns(5)
    for i, skill in enumerate(skills):
        with cols[i % 5]:
            if st.button(skill):
                st.session_state.selected_skill = skill
                st.rerun()
    
    # Show total number of job postings
    st.subheader("📁 Dataset Summary")
    st.metric("Total Job Postings", f"{len(df):,}")

# --- PAGE 2: Skill Details ---
else:
    selected_skill = st.session_state.selected_skill

    # Back button
    if st.button("⬅️ Back to Top Skills"):
        st.session_state.selected_skill = None
        st.rerun()

    st.markdown(f"### 🔍 {selected_skill} analytics")
    st.caption(f"Detailed breakdown of job postings mentioning **{selected_skill}**.")

    # Filter for selected skill
    df_skill = df[df[skills_col].apply(lambda x: selected_skill in x if isinstance(x, list) else False)]

    # --- Key metrics ---
    total_postings = len(df_skill)
    top_category = df_skill["cluster_label"].mode()[0] if "cluster_label" in df_skill else "N/A"
    top_location = df_skill["location_cleaned"].mode()[0] if "location_cleaned" in df_skill else "N/A"

    c1, c2 = st.columns(2)
    c1.metric("Total Postings", f"{total_postings:,}")
    c2.metric("Top Location", top_location)

    st.metric("Top Category", top_category)

    # --- Category Distribution ---
    st.subheader(f"Top Sectors for {selected_skill}")
    cat_counts = df_skill["cluster_label"].value_counts().head(5) 

    # convert Series -> DataFrame with explicit column names
    cat_df = cat_counts.reset_index()
    cat_df.columns = ["Category", "Count"]

    # Atlair bar chart
    chart = (
        alt.Chart(cat_df)
        .mark_bar()
        .encode(
            y=alt.Y("Category:N", sort="-x", title="Category"),
            x=alt.X("Count:Q", title="Job Postings"),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="tealblues"), legend=None),
            tooltip=["Category", "Count"]
        )
        .properties(width=900, height=500)
    )
    st.altair_chart(chart, use_container_width=True)




