"""Trends - Historical progress on sex metadata completeness."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sage.database import fetch_snapshots, get_filter_options, get_disease_categories

st.set_page_config(page_title="Trends | SAGE", page_icon="📈", layout="wide")

st.title("📈 Trends")
st.markdown(
    "Track historical progress in sex metadata completeness across the genomics community. "
    "Snapshots are captured weekly to measure improvement over time."
)

st.divider()

# Filters section
st.subheader("Filters")

col1, col2 = st.columns(2)

try:
    filter_options = get_filter_options()
    organisms = filter_options.get("organisms", [])
    disease_categories = get_disease_categories()

    with col1:
        selected_organism = st.selectbox(
            "Filter by organism",
            ["All"] + organisms,
            index=0,
        )
        organism_filter = None if selected_organism == "All" else selected_organism

    with col2:
        selected_disease = st.selectbox(
            "Filter by disease category",
            ["All"] + disease_categories,
            index=0,
        )
        disease_filter = None if selected_disease == "All" else selected_disease

except Exception as e:
    st.error(f"Error loading filters: {e}")
    organism_filter = None
    disease_filter = None

st.divider()

# Fetch snapshots
try:
    snapshots = fetch_snapshots(organism=organism_filter, disease_category=disease_filter, limit=260)

    if not snapshots or len(snapshots) < 2:
        st.warning("Not enough historical data yet. Snapshots are captured weekly.")
        st.info(
            "To create snapshots manually: `poetry run python scripts/create_snapshot.py`\n\n"
            "Weekly snapshots will automatically start once the GitHub Actions workflow is enabled."
        )
        st.stop()

    # Convert to DataFrame
    df = pd.DataFrame(snapshots)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values("snapshot_date")

    # Section 1: Overall Completeness Trend
    st.subheader("Sex Metadata Completeness Over Time")
    st.markdown(
        "Percentage of studies with at least some sex metadata documented. " "Higher is better!"
    )

    completeness_data = df.copy()
    completeness_data["completeness_pct"] = completeness_data["avg_metadata_completeness"] * 100

    fig_completeness = px.line(
        completeness_data,
        x="snapshot_date",
        y="completeness_pct",
        markers=True,
        labels={
            "snapshot_date": "Date",
            "completeness_pct": "Avg Sex Metadata %",
        },
        color_discrete_sequence=["#1f77b4"],
    )
    fig_completeness.update_traces(
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Completeness: %{y:.1f}%<extra></extra>"
    )
    fig_completeness.update_layout(
        height=400,
        hovermode="x unified",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_completeness, use_container_width=True)

    # Section 2: Study Counts Over Time
    st.subheader("Study Growth")
    st.markdown(
        "Total studies and breakdown by metadata completeness status. "
        "Watch how the field improves!"
    )

    counts_data = df.copy()
    counts_data["studies_without_metadata"] = (
        counts_data["total_studies"] - counts_data["studies_with_sex_metadata"]
    )

    # Create stacked area chart
    fig_growth = go.Figure()

    fig_growth.add_trace(
        go.Scatter(
            x=counts_data["snapshot_date"],
            y=counts_data["studies_with_sex_metadata"],
            name="With Sex Metadata",
            mode="lines",
            line=dict(width=0.5, color="rgb(34, 139, 34)"),
            fillcolor="rgba(34, 139, 34, 0.4)",
            fill="tonexty",
        )
    )

    fig_growth.add_trace(
        go.Scatter(
            x=counts_data["snapshot_date"],
            y=counts_data["studies_without_metadata"],
            name="Missing Sex Metadata",
            mode="lines",
            line=dict(width=0.5, color="rgb(220, 20, 60)"),
            fillcolor="rgba(220, 20, 60, 0.4)",
            fill="tozeroy",
        )
    )

    fig_growth.update_layout(
        title="",
        xaxis_title="Date",
        yaxis_title="Study Count",
        hovermode="x unified",
        height=400,
        legend=dict(x=0.01, y=0.99),
    )

    st.plotly_chart(fig_growth, use_container_width=True)

    # Section 3: Weekly Change Rate
    st.subheader("Recent Progress")

    if len(df) >= 2:
        latest = df.iloc[-1]
        previous = df.iloc[-2]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_studies = latest["total_studies"] - previous["total_studies"]
            st.metric(
                label="New Studies",
                value=f"{latest['total_studies']:,}",
                delta=f"{delta_studies:+.0f}",
                delta_color="normal",
            )

        with col2:
            delta_with_sex = (
                latest["studies_with_sex_metadata"] - previous["studies_with_sex_metadata"]
            )
            st.metric(
                label="With Sex Metadata",
                value=f"{latest['studies_with_sex_metadata']:,}",
                delta=f"{delta_with_sex:+.0f}",
                delta_color="normal",
            )

        with col3:
            latest_pct = latest["avg_metadata_completeness"] * 100
            previous_pct = previous["avg_metadata_completeness"] * 100
            delta_pct = latest_pct - previous_pct
            st.metric(
                label="Avg Completeness %",
                value=f"{latest_pct:.1f}%",
                delta=f"{delta_pct:+.1f}%",
                delta_color="off" if delta_pct == 0 else ("normal" if delta_pct > 0 else "inverse"),
            )

        with col4:
            inferrable_pct = (
                (latest["studies_sex_inferrable"] / latest["total_studies"] * 100)
                if latest["total_studies"] > 0
                else 0
            )
            st.metric(
                label="Sex Inferrable %",
                value=f"{inferrable_pct:.1f}%",
            )
    else:
        st.info("Need at least 2 snapshots to calculate changes")

    st.divider()

    # Section 4: Detailed Snapshot Table
    st.subheader("Snapshot History")

    display_data = []
    for _, row in df.iterrows():
        completeness_pct = row["avg_metadata_completeness"] * 100
        inferrable_pct = (
            (row["studies_sex_inferrable"] / row["total_studies"] * 100)
            if row["total_studies"] > 0
            else 0
        )
        analyzed_pct = (
            (row["studies_with_sex_analysis"] / row["total_studies"] * 100)
            if row["total_studies"] > 0
            else 0
        )

        display_data.append(
            {
                "Date": row["snapshot_date"].strftime("%Y-%m-%d"),
                "Total Studies": f"{row['total_studies']:,}",
                "With Metadata": f"{row['studies_with_sex_metadata']:,}",
                "Inferrable": f"{row['studies_sex_inferrable']:,}",
                "Analyzed": f"{row['studies_with_sex_analysis']:,}",
                "Avg Completeness %": f"{completeness_pct:.1f}%",
                "Inferrable %": f"{inferrable_pct:.1f}%",
                "Analyzed %": f"{analyzed_pct:.1f}%",
            }
        )

    df_display = pd.DataFrame(display_data)
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    st.divider()
    st.subheader("Summary Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_improvement = (
            df.iloc[-1]["avg_metadata_completeness"] - df.iloc[0]["avg_metadata_completeness"]
        ) * 100
        st.metric(
            label="Total Completeness Improvement",
            value=f"{total_improvement:.1f}%",
        )

    with col2:
        study_growth = df.iloc[-1]["total_studies"] - df.iloc[0]["total_studies"]
        st.metric(
            label="Study Growth",
            value=f"{study_growth:,}",
        )

    with col3:
        avg_completeness = df["avg_metadata_completeness"].mean() * 100
        st.metric(
            label="Average Completeness (All Time)",
            value=f"{avg_completeness:.1f}%",
        )

except Exception as e:
    st.error(f"Error loading trend data: {e}")
    st.info("Make sure your Supabase connection is configured correctly.")
