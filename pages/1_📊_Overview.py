"""Overview page - Summary statistics on sex metadata completeness."""
import pandas as pd
import plotly.express as px
import streamlit as st

from sage.metrics import calculate_completeness_percentage

st.set_page_config(page_title="Overview | SAGE", page_icon="📊", layout="wide")

st.title("📊 Overview")
st.markdown("Summary statistics on sex metadata completeness across public genomics studies.")

# Mock data for MVP
stats = {
    "total_studies": 127432,
    "with_sex_metadata": 87123,
    "sex_inferrable": 23891,
    "with_sex_analysis": 15734,
}

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

# Mock data for organism breakdown
organism_data = pd.DataFrame(
    {
        "Organism": ["Homo sapiens", "Mus musculus", "Rattus norvegicus", "Danio rerio"],
        "Total Studies": [89234, 28123, 7834, 2241],
        "With Sex Metadata": [72145, 15234, 3456, 1123],
        "Completeness %": [80.9, 54.1, 44.1, 50.1],
    }
)

st.subheader("Completeness by Organism")
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
st.plotly_chart(fig_organism, use_container_width=True)

# Mock data for disease category
disease_data = pd.DataFrame(
    {
        "Disease Category": [
            "Cancer",
            "Neurological",
            "Cardiovascular",
            "Autoimmune",
            "Metabolic",
            "Infectious",
        ],
        "Studies": [34567, 12345, 8234, 5123, 6789, 4321],
        "Completeness %": [82.3, 67.2, 54.1, 62.3, 58.9, 71.2],
    }
)

st.subheader("Completeness by Disease Category")
fig_disease = px.bar(
    disease_data,
    x="Disease Category",
    y="Completeness %",
    color="Completeness %",
    color_continuous_scale="RdYlGn",
    range_color=[0, 100],
    text="Completeness %",
)
fig_disease.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
st.plotly_chart(fig_disease, use_container_width=True)

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
