"""Disease Explorer - Browse diseases by sex metadata completeness."""
import pandas as pd
import plotly.express as px
import streamlit as st

from sage.database import (
    fetch_disease_stats,
    get_diseases_with_completeness,
    get_studies_for_disease,
    get_disease_categories,
)

st.set_page_config(page_title="Disease Explorer | SAGE", page_icon="🔬", layout="wide")

st.title("🔬 Disease Explorer")
st.markdown(
    "Browse diseases by study count and sex metadata completeness to identify "
    "high-value opportunities for sex-aware reanalysis."
)

st.divider()

# Initialize session state for filters
if "disease_category_filter" not in st.session_state:
    st.session_state.disease_category_filter = None
if "min_study_count_filter" not in st.session_state:
    st.session_state.min_study_count_filter = 1
if "known_sex_diff_filter" not in st.session_state:
    st.session_state.known_sex_diff_filter = False
if "selected_disease_detail" not in st.session_state:
    st.session_state.selected_disease_detail = None

# Display header metrics
try:
    stats = fetch_disease_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Total Diseases", value=f"{stats['total_diseases']:,}")

    with col2:
        st.metric(
            label="Diseases with Studies",
            value=f"{stats['diseases_with_studies']:,}",
        )

    with col3:
        st.metric(
            label="Avg Sex Metadata %",
            value=f"{stats['avg_completeness']:.1f}%",
        )

    with col4:
        st.metric(
            label="Study Mappings",
            value=f"{stats['total_study_mappings']:,}",
        )

except Exception as e:
    st.error(f"Error fetching disease statistics: {e}")
    st.info("Make sure your Supabase connection is configured correctly.")

st.divider()

# Filters section
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

try:
    disease_categories = get_disease_categories()

    with col1:
        selected_category = st.selectbox(
            "Filter by disease category",
            ["All"] + disease_categories,
            index=0,
        )
        st.session_state.disease_category_filter = (
            None if selected_category == "All" else selected_category
        )

    with col2:
        min_studies = st.slider(
            "Minimum study count",
            min_value=1,
            max_value=20,
            value=1,
        )
        st.session_state.min_study_count_filter = min_studies

    with col3:
        known_diff_only = st.checkbox("Only show diseases with known sex differences")
        st.session_state.known_sex_diff_filter = known_diff_only

except Exception as e:
    st.error(f"Error loading filters: {e}")
    disease_categories = []

st.divider()

# Main content
try:
    # Fetch diseases with filters
    diseases = get_diseases_with_completeness(
        disease_category=st.session_state.disease_category_filter,
        min_studies=st.session_state.min_study_count_filter,
        known_sex_diff_only=st.session_state.known_sex_diff_filter,
        limit=50,
    )

    if diseases:
        # Section 1: Visualization
        st.subheader("Sex Metadata Completeness by Disease")

        df_viz = pd.DataFrame(diseases[:20])  # Show top 20 for readability

        fig = px.bar(
            df_viz,
            x="disease_term",
            y="avg_completeness",
            color="study_count",
            hover_data=["disease_category", "known_sex_difference"],
            labels={
                "disease_term": "Disease",
                "avg_completeness": "Sex Metadata Completeness (%)",
                "study_count": "Study Count",
            },
            color_continuous_scale="Viridis",
        )
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, width="stretch")

        st.info(f"Showing top 20 of {len(diseases)} diseases")

        st.divider()

        # Section 2: Disease details table
        st.subheader("Disease Details")

        display_data = [
            {
                "Disease": d["disease_term"],
                "Category": d["disease_category"] or "Unknown",
                "Studies": d["study_count"],
                "Avg Completeness %": f"{d['avg_completeness']:.1f}%",
                "Known Sex Diff": "✓" if d.get("known_sex_difference") else "✗",
                "Priority Score": f"{d.get('avg_clinical_priority', 0):.2f}",
            }
            for d in diseases
        ]

        df_diseases = pd.DataFrame(display_data)

        st.dataframe(
            df_diseases,
            width="stretch",
            hide_index=True,
        )

        st.divider()

        # Section 3: Drill-down functionality
        st.subheader("View Studies for a Disease")

        disease_options = [d["disease_term"] for d in diseases]
        selected_disease = st.selectbox(
            "Select a disease to view its studies",
            [""] + disease_options,
        )

        if selected_disease:
            try:
                studies = get_studies_for_disease(selected_disease, limit=100)

                if studies:
                    st.success(f"Found {len(studies)} studies for **{selected_disease}**")

                    # Display studies table
                    studies_display = [
                        {
                            "Accession": s.get("geo_accession", ""),
                            "Title": s.get("title", "")[:60]
                            + ("..." if len(s.get("title", "")) > 60 else ""),
                            "Organism": s.get("organism", ""),
                            "Samples": s.get("sample_count", 0),
                            "Sex Metadata %": f"{(s.get('sex_metadata_completeness', 0) or 0) * 100:.1f}%",
                            "Has Analysis": ("✓" if s.get("reports_sex_analysis") else "✗"),
                        }
                        for s in studies
                    ]

                    df_studies = pd.DataFrame(studies_display)
                    st.dataframe(df_studies, width="stretch", hide_index=True)
                else:
                    st.info(f"No studies found for **{selected_disease}**")

            except Exception as e:
                st.error(f"Error loading studies: {e}")

    else:
        st.info("No diseases found matching your filters. Try adjusting them.")

except Exception as e:
    st.error(f"Error loading disease data: {e}")
    st.info("Make sure your Supabase connection is configured correctly.")
