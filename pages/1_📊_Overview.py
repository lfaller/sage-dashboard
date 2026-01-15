"""Overview page - Summary statistics on sex metadata completeness."""
import pandas as pd
import plotly.express as px
import streamlit as st

from sage.metrics import calculate_completeness_percentage
from sage.database import fetch_overview_stats, search_studies

st.set_page_config(page_title="Overview | SAGE", page_icon="📊", layout="wide")

st.title("📊 Overview")
st.markdown("Summary statistics on sex metadata completeness across public genomics studies.")

# Fetch real data from Supabase
stats = fetch_overview_stats()

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Studies", value=f"{stats['total_studies']:,}")

with col2:
    pct_with_sex = calculate_completeness_percentage(
        stats["with_sex_metadata"], stats["total_studies"]
    )
    st.metric(
        label="Have Sex Metadata",
        value=f"{pct_with_sex:.1f}%",
        delta=f"{stats['with_sex_metadata']:,} studies",
    )

with col3:
    st.metric(label="Rescuable via Inference", value=f"{stats['sex_inferrable']:,}")

with col4:
    pct_analyzed = calculate_completeness_percentage(
        stats["with_sex_analysis"], stats["total_studies"]
    )
    st.metric(
        label="Analyzed by Sex",
        value=f"{pct_analyzed:.1f}%",
        delta=f"{stats['with_sex_analysis']:,} studies",
    )

st.divider()

# Fetch organism breakdown from database
st.subheader("Completeness by Organism")
try:
    all_studies = search_studies(limit=1000)

    if all_studies:
        # Group studies by organism
        organism_stats = {}
        for study in all_studies:
            organism = study.get("organism", "Unknown")
            if organism not in organism_stats:
                organism_stats[organism] = {
                    "total": 0,
                    "with_sex": 0,
                }
            organism_stats[organism]["total"] += 1
            if study.get("has_sex_metadata", False):
                organism_stats[organism]["with_sex"] += 1

        # Filter to only human and mouse studies
        organism_mapping = {
            "Homo sapiens": "Human",
            "Mus musculus": "Mouse",
        }

        filtered_stats = {
            org: stats for org, stats in organism_stats.items() if org in organism_mapping
        }

        # Create dataframe
        organism_data = pd.DataFrame(
            [
                {
                    "Organism": organism_mapping.get(organism, organism),
                    "Total Studies": stats["total"],
                    "With Sex Metadata": stats["with_sex"],
                    "Completeness %": (stats["with_sex"] / stats["total"] * 100)
                    if stats["total"] > 0
                    else 0,
                }
                for organism, stats in filtered_stats.items()
            ]
        ).sort_values("Completeness %", ascending=False)

        fig_organism = px.bar(
            organism_data,
            x="Organism",
            y="Completeness %",
            color="Completeness %",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            text="Completeness %",
        )
        fig_organism.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig_organism, width="stretch")
    else:
        st.info("No studies found in database")
except Exception as e:
    st.error(f"Error fetching organism data: {e}")
    st.info("Make sure your Supabase connection is configured.")

st.subheader("Study Type Distribution")
try:
    # Filter to only RNA-seq studies (primary focus for sex metadata)
    rna_seq_studies = [s for s in all_studies if s.get("study_type") == "RNA-seq"]

    if rna_seq_studies:
        rna_seq_with_sex = sum(1 for s in rna_seq_studies if s.get("has_sex_metadata", False))
        rna_seq_total = len(rna_seq_studies)
        rna_seq_pct = (
            (rna_seq_with_sex / rna_seq_total * 100) if rna_seq_total > 0 else 0
        )

        st.metric(
            label="RNA-seq Studies with Sex Metadata",
            value=f"{rna_seq_with_sex}/{rna_seq_total}",
            delta=f"{rna_seq_pct:.1f}%",
        )
    else:
        st.info("No RNA-seq studies found in database")

except Exception as e:
    st.error(f"Error fetching study type data: {e}")
    st.info("Make sure your Supabase connection is configured.")

st.divider()

st.subheader("Summary Table")
summary_df = pd.DataFrame(
    {
        "Metric": [
            "Total Studies Analyzed",
            "Studies with Sex Metadata",
            "Missing Sex Metadata",
            "Inferrable Studies",
            "Analyzed by Sex",
        ],
        "Count": [
            f"{stats['total_studies']:,}",
            f"{stats['with_sex_metadata']:,}",
            f"{stats['total_studies'] - stats['with_sex_metadata']:,}",
            f"{stats['sex_inferrable']:,}",
            f"{stats['with_sex_analysis']:,}",
        ],
        "Percentage": [
            "100%",
            f"{calculate_completeness_percentage(stats['with_sex_metadata'], stats['total_studies']):.1f}%",
            f"{calculate_completeness_percentage(stats['total_studies'] - stats['with_sex_metadata'], stats['total_studies']):.1f}%",
            f"{calculate_completeness_percentage(stats['sex_inferrable'], stats['total_studies']):.1f}%",
            f"{calculate_completeness_percentage(stats['with_sex_analysis'], stats['total_studies']):.1f}%",
        ],
    }
)
st.table(summary_df)
