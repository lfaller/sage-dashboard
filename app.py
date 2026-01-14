"""SAGE Dashboard - Main entry point."""
import streamlit as st

st.set_page_config(
    page_title="SAGE Dashboard", page_icon="🧬", layout="wide", initial_sidebar_state="expanded"
)

st.title("🧬 SAGE Dashboard")
st.subheader("Sex-Aware Genomics Explorer")

st.markdown(
    """
**SAGE** illuminates the gaps in sex-stratified genomics data, helping researchers
identify opportunities to improve metadata quality and enable sex-aware analysis.

### Why This Matters

- **12.5%** of human genes show sex-biased expression
- **25%+** of critical metadata is missing from public studies
- **22%** of significant expression differences are hidden when not stratifying by sex
- Women experience adverse drug reactions **2x** as often as men

### Current Features

👈 Use the sidebar to explore:
- 📊 **Overview** - Summary statistics on sex metadata completeness
- 🔍 **Study Search** - Find and filter studies by organism, study type, and sex metadata
- 🔬 **Disease Explorer** - Browse diseases and identify high-value reanalysis opportunities
"""
)

st.divider()

st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <p>Built by <a href='https://linafaller.com'>Lina Faller</a> |
    <a href='https://github.com/linafaller/sage-dashboard'>GitHub</a></p>
</div>
""",
    unsafe_allow_html=True,
)
