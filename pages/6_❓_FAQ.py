"""FAQ - Methodology and common questions about the Trends page."""
import streamlit as st

st.set_page_config(page_title="FAQ | SAGE", page_icon="❓", layout="wide")

st.title("❓ FAQ")
st.markdown(
    "Common questions about how SAGE measures sex metadata completeness and calculates trends."
)

st.divider()

# Sex metadata detection
with st.expander("How does SAGE determine if a study has sex metadata?", expanded=True):
    st.markdown(
        """
    SAGE analyzes **sample names/titles** in published GEO studies to detect explicit sex labels. We scan for:

    - **M/F patterns**: Labels like `M1`, `M2`, `F1`, `F2` (M or F followed by a number, dash, dot, or underscore)
    - **Full words**: `Male`, `Female` (case-insensitive)

    For example:
    - ✅ Sample name `Male_001_WT` → detected
    - ✅ Sample name `F2_Treatment` → detected
    - ❌ Sample name `GSM123456_WT` → not detected (no sex label)

    **Important limitation**: We only analyze sample names/titles available in GEO metadata. We do NOT download
    and inspect full sample characteristics or expression data.
    """
    )

# Completeness metric
with st.expander("What does 'Sex Metadata Completeness %' mean?"):
    st.markdown(
        """
    This metric shows the **percentage of samples with explicit sex labels** in a study:

    - **100%**: All samples have sex labels (e.g., every sample name starts with M or F)
    - **50%**: Half the samples have sex labels
    - **0%**: No samples have sex labels

    The Trends page shows the **average completeness across all studies** in each snapshot, helping visualize
    how well the field documents sex information in study samples.
    """
    )

# Metrics explanation
with st.expander("What are the different metrics in the Trends view?"):
    st.markdown(
        """
    #### Total Studies
    The total count of studies in the database at that snapshot date. This grows over time as new studies are
    published and added to SAGE.

    #### Studies with Sex Metadata
    Studies where at least one sample has an explicit sex label (sex metadata completeness > 0%). This shows
    how many studies are contributing sex information to the field.

    #### Sex Inferrable Studies
    Studies where sex could potentially be inferred from gene expression data, based on a confidence score.
    This requires:
    - RNA-seq study type (35% confidence)
    - Sample size ≥ 20 samples (25% confidence)
    - Human organism (15% confidence)
    - Sample name sex labels (up to 25% confidence)

    Studies scoring ≥ 50% confidence are marked as "sex inferrable."

    #### Studies with Sex Analysis
    Studies that explicitly report sex-specific analysis in their publication. This is the gold standard—studies
    where researchers actively analyzed differences between sexes.
    """
    )

# Snapshot creation
with st.expander("How are snapshots created?"):
    st.markdown(
        """
    **Weekly Snapshots**: Every Monday at 00:00 UTC, SAGE captures a snapshot of current metrics across all
    studies in the database.

    **Historical Snapshots**: To enable the 5-year trend view, historical snapshots are retroactively created.
    For a snapshot dated January 1, 2021, we include all studies with a publication date on or before
    January 1, 2021. This simulates what the metrics would have looked like on that date without waiting
    years for real snapshots.
    """
    )

# Using the Trends page
with st.expander("How can I use the Trends page to understand progress?"):
    st.markdown(
        """
    The Trends page shows:

    1. **Overall completeness trend** (top chart): Has the field improved at documenting sex information
       in sample labels?
    2. **Study growth** (middle chart): Are more studies being published? Are they getting better at
       including sex metadata?
    3. **Recent progress** (bottom left): What changed in the last week?
    4. **Summary statistics** (bottom right): Overall improvement from the earliest to most recent snapshot

    Use the filters to focus on:
    - **Specific organisms** (e.g., "Homo sapiens" for human studies)
    - **Disease categories** (e.g., cancer, cardiovascular) to see which areas are improving
    """
    )

# No data issues
with st.expander("Why does my filter selection show no data?"):
    st.markdown(
        """
    Some combinations of organism + disease category may have very few studies. If you see
    "Not enough historical data yet," try:
    - Selecting "All" for one or both filters to see the overall trend
    - Checking a broader disease category
    - Ensuring the date range captured includes your filtered studies
    """
    )

# Snapshot frequency
with st.expander("How often are snapshots captured?"):
    st.markdown(
        """
    - **Weekly**: Automated snapshots every Monday at 00:00 UTC
    - **Historical**: One-time backfill captures weekly data going back 5 years
    - **Manual**: You can create on-demand snapshots using `poetry run python scripts/create_snapshot.py`
    """
    )

# Data sources
with st.expander("What data sources does SAGE use?"):
    st.markdown(
        """
    SAGE analyzes studies from **NCBI GEO (Gene Expression Omnibus)**, specifically:
    - RNA-seq studies
    - Human and model organism studies
    - Studies with sample-level metadata available

    We do NOT modify or reanalyze the underlying expression data—we only catalog and analyze the metadata
    that researchers provide.
    """
    )

# Improving metadata
with st.expander("How can studies improve their sex metadata completeness?"):
    st.markdown(
        """
    When publishing to GEO, researchers should:

    1. **Include sex in sample labels**: Use consistent naming like `M1`, `F1` or `Male_001`, `Female_001`
    2. **Document sex in sample characteristics**: Include a "sex" or "gender" field in the sample metadata
    3. **Report sex-specific analysis**: If sex differences exist, highlight them in the publication
    4. **Use standardized vocabulary**: Follow GEO guidelines for consistent metadata formatting
    """
    )

# Download data
with st.expander("Can I download the snapshot data?"):
    st.markdown(
        """
    Not yet from the UI, but you can:
    - Query the `completeness_snapshots` table directly in Supabase
    - Use the internal `fetch_snapshots()` function from the database module

    Contact the SAGE team if you need raw data for analysis.
    """
    )

# Detection accuracy
with st.expander("How accurate is the sex metadata detection?"):
    st.markdown(
        """
    Sex label detection using sample names is **highly accurate** for explicit patterns (M/F, Male/Female),
    but has limitations:

    - **Cannot detect**: Undocumented sex info, coded values (e.g., "1" for male), non-English labels
    - **False positives**: Rare but possible (e.g., a sample named "FM123" might be misinterpreted)

    For comprehensive analysis, check the study's full metadata in GEO and the publication itself.
    """
    )

# Contributing
with st.expander("How can I contribute to SAGE?"):
    st.markdown(
        """
    Visit the [GitHub repository](https://github.com/lfaller/sage-dashboard) to:
    - Report issues with metadata detection
    - Suggest improvements to the Trends methodology
    - Contribute code or documentation
    """
    )
