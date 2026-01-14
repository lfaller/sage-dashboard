"""Rescue Finder - Identify high-value datasets for sex inference rescue."""
import pandas as pd
import plotly.express as px
import streamlit as st

from sage.database import (
    get_rescue_opportunities,
    fetch_rescue_stats,
    get_filter_options,
    get_disease_categories,
)

st.set_page_config(page_title="Rescue Finder | SAGE", page_icon="🎯", layout="wide")

st.title("🎯 Rescue Finder")
st.markdown(
    "Identify studies lacking sex metadata where sex can be computationally inferred "
    "from sample characteristics and naming patterns. Ranked by rescue potential score."
)

st.divider()

# Initialize session state for filters
if "rescue_organism_filter" not in st.session_state:
    st.session_state.rescue_organism_filter = None
if "rescue_disease_filter" not in st.session_state:
    st.session_state.rescue_disease_filter = None
if "rescue_min_confidence" not in st.session_state:
    st.session_state.rescue_min_confidence = 0.0
if "rescue_selected_study" not in st.session_state:
    st.session_state.rescue_selected_study = None

# Display header metrics
try:
    stats = fetch_rescue_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Rescue Opportunities",
            value=f"{stats['total_opportunities']:,}",
        )

    with col2:
        st.metric(
            label="High Confidence (≥0.7)",
            value=f"{stats['high_confidence_count']:,}",
        )

    with col3:
        st.metric(
            label="Potential Samples",
            value=f"{stats['potential_samples']:,}",
        )

    with col4:
        if stats["total_opportunities"] > 0:
            rescue_rate = stats["high_confidence_count"] / stats["total_opportunities"] * 100
            st.metric(label="High Confidence Rate", value=f"{rescue_rate:.1f}%")
        else:
            st.metric(label="High Confidence Rate", value="0%")

except Exception as e:
    st.error(f"Error fetching rescue statistics: {e}")
    st.info("Make sure your Supabase connection is configured correctly.")

st.divider()

# Filters section
st.subheader("Filters")

col1, col2, col3 = st.columns(3)

try:
    organisms = get_filter_options()["organisms"]
    disease_categories = get_disease_categories()

    with col1:
        selected_organism = st.selectbox(
            "Filter by organism",
            ["All"] + organisms,
            index=0,
        )
        st.session_state.rescue_organism_filter = (
            None if selected_organism == "All" else selected_organism
        )

    with col2:
        selected_disease = st.selectbox(
            "Filter by disease category",
            ["All"] + disease_categories,
            index=0,
        )
        st.session_state.rescue_disease_filter = (
            None if selected_disease == "All" else selected_disease
        )

    with col3:
        min_confidence = st.slider(
            "Minimum inference confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )
        st.session_state.rescue_min_confidence = min_confidence

except Exception as e:
    st.error(f"Error loading filters: {e}")
    organisms = []
    disease_categories = []

st.divider()

# Main content
try:
    # Fetch rescue opportunities with filters
    opportunities = get_rescue_opportunities(
        organism=st.session_state.rescue_organism_filter,
        disease_category=st.session_state.rescue_disease_filter,
        min_confidence=st.session_state.rescue_min_confidence,
        limit=100,
    )

    if opportunities:
        # Section 1: Visualization
        st.subheader("Rescue Potential by Study")

        df_viz = pd.DataFrame(opportunities[:20])  # Show top 20 for readability

        fig = px.bar(
            df_viz,
            x="geo_accession",
            y="rescue_score",
            color="sex_inference_confidence",
            hover_data=[
                "title",
                "organism",
                "study_type",
                "sample_count",
                "rescue_score",
            ],
            labels={
                "geo_accession": "Study Accession",
                "rescue_score": "Rescue Score",
                "sex_inference_confidence": "Inference Confidence",
            },
            color_continuous_scale="Viridis",
        )
        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"Showing top 20 of {len(opportunities)} rescue opportunities")

        st.divider()

        # Section 2: Ranked opportunities table
        st.subheader("Ranked Opportunities")

        display_data = [
            {
                "Study": o.get("geo_accession", ""),
                "Type": o.get("study_type", ""),
                "Samples": o.get("sample_count", 0),
                "Organism": o.get("organism", ""),
                "Inference Confidence": f"{o.get('sex_inference_confidence', 0):.2f}",
                "Rescue Score": f"{o.get('rescue_score', 0):.3f}",
            }
            for o in opportunities
        ]

        df_opps = pd.DataFrame(display_data)

        st.dataframe(
            df_opps,
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Section 3: Drill-down functionality
        st.subheader("Study Details")

        study_options = [
            f"{o['geo_accession']} - {o['title'][:40]}..."
            if len(o.get("title", "")) > 40
            else f"{o['geo_accession']} - {o.get('title', '')}"
            for o in opportunities
        ]
        selected_study_option = st.selectbox(
            "Select a study to view detailed inference analysis",
            [""] + study_options,
        )

        if selected_study_option:
            # Extract the accession from the display string
            accession = selected_study_option.split(" - ")[0]

            # Find the full study record
            selected_study = next(
                (o for o in opportunities if o["geo_accession"] == accession),
                None,
            )

            if selected_study:
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.success(f"**{selected_study['geo_accession']}**")
                    st.write(f"**Title:** {selected_study.get('title', 'N/A')}")
                    st.write(f"**Organism:** {selected_study.get('organism', 'N/A')}")
                    st.write(f"**Study Type:** {selected_study.get('study_type', 'N/A')}")
                    st.write(f"**Sample Count:** {selected_study.get('sample_count', 0)}")

                with col2:
                    st.metric(
                        "Inference Confidence",
                        f"{selected_study.get('sex_inference_confidence', 0):.2f}",
                    )
                    st.metric(
                        "Rescue Score",
                        f"{selected_study.get('rescue_score', 0):.3f}",
                    )
                    st.metric(
                        "Sex Metadata %",
                        f"{selected_study.get('sex_metadata_completeness', 0) * 100:.1f}%",
                    )

                st.divider()

                # Inference details
                st.subheader("Inference Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Inference Factors:**")
                    st.write(
                        f"• RNA-seq Study: {'Yes' if selected_study.get('study_type') == 'RNA-seq' else 'No'}"
                    )
                    st.write(
                        f"• Sample Count ≥20: {'Yes' if selected_study.get('sample_count', 0) >= 20 else 'No'}"
                    )
                    st.write(
                        f"• Human Organism: {'Yes' if selected_study.get('organism') == 'Homo sapiens' else 'No'}"
                    )

                with col2:
                    st.write("**Score Composition (weighted):**")
                    st.write("• Inference Confidence: 30%")
                    st.write("• Sample Size: 25%")
                    st.write("• Missing Metadata: 20%")
                    st.write("• Study Type: 15%")
                    st.write("• Clinical Priority: 10%")

                st.divider()

                # GEO link
                geo_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
                st.markdown(
                    f"📖 [View on GEO Database]({geo_url})",
                    help="Open this study on the GEO database website",
                )

    else:
        st.info("No rescue opportunities found matching your filters. Try adjusting them.")

except Exception as e:
    st.error(f"Error loading rescue opportunities: {e}")
    st.info("Make sure your Supabase connection is configured correctly.")
