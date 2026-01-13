"""Study search and filtering page."""
import streamlit as st
import pandas as pd

from sage.database import search_studies_advanced, get_filter_options

st.set_page_config(page_title="Study Search | SAGE", page_icon="🔍", layout="wide")

st.title("🔍 Study Search")
st.markdown("Find and filter studies by organism, study type, and sex metadata characteristics.")

st.divider()

# Initialize session state for search
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "selected_organism" not in st.session_state:
    st.session_state.selected_organism = None
if "selected_study_type" not in st.session_state:
    st.session_state.selected_study_type = None
if "has_sex_meta_filter" not in st.session_state:
    st.session_state.has_sex_meta_filter = None
if "page_number" not in st.session_state:
    st.session_state.page_number = 0

# Search and filter controls
with st.container():
    st.subheader("Search & Filter")

    col1, col2, col3 = st.columns(3)

    with col1:
        search_query = st.text_input(
            "Search studies by title or accession",
            value=st.session_state.search_query,
            placeholder="e.g., breast cancer, GSE123001",
        )
        st.session_state.search_query = search_query

    # Get filter options
    try:
        options = get_filter_options()
        organisms = options.get("organisms", [])
        study_types = options.get("study_types", [])
    except Exception as e:
        st.error(f"Error loading filter options: {e}")
        organisms = []
        study_types = []

    with col2:
        selected_organism = st.selectbox(
            "Filter by organism",
            ["All"] + organisms,
            index=0
            if st.session_state.selected_organism is None
            else (
                organisms.index(st.session_state.selected_organism) + 1
                if st.session_state.selected_organism in organisms
                else 0
            ),
        )
        st.session_state.selected_organism = (
            None if selected_organism == "All" else selected_organism
        )

    with col3:
        selected_study_type = st.selectbox(
            "Filter by study type",
            ["All"] + study_types,
            index=0
            if st.session_state.selected_study_type is None
            else (
                study_types.index(st.session_state.selected_study_type) + 1
                if st.session_state.selected_study_type in study_types
                else 0
            ),
        )
        st.session_state.selected_study_type = (
            None if selected_study_type == "All" else selected_study_type
        )

    col4, col5 = st.columns([2, 1])

    with col4:
        has_sex_metadata = st.checkbox("Only show studies with sex metadata")
        st.session_state.has_sex_meta_filter = True if has_sex_metadata else None

    with col5:
        if st.button("Clear Filters", use_container_width=True):
            st.session_state.search_query = ""
            st.session_state.selected_organism = None
            st.session_state.selected_study_type = None
            st.session_state.has_sex_meta_filter = None
            st.session_state.page_number = 0
            st.rerun()

st.divider()

# Perform search
try:
    result = search_studies_advanced(
        search=st.session_state.search_query if st.session_state.search_query else None,
        organism=st.session_state.selected_organism,
        study_type=st.session_state.selected_study_type,
        has_sex_metadata=st.session_state.has_sex_meta_filter,
        limit=50,
        offset=st.session_state.page_number * 50,
    )

    total_results = result["total"]
    studies = result["results"]

    # Display results info
    if total_results > 0:
        st.success(f"Found {total_results} studies")
    else:
        st.info("No studies found. Try adjusting your search criteria.")

    # Display results table
    if studies:
        # Prepare data for display
        display_data = []
        for study in studies:
            completeness = study.get("sex_metadata_completeness", 0) or 0
            completeness_pct = f"{completeness * 100:.1f}%"
            inferrable_status = "✓" if study.get("sex_inferrable") else "✗"

            display_data.append(
                {
                    "Accession": study.get("geo_accession", ""),
                    "Title": study.get("title", "")[:60]
                    + ("..." if len(study.get("title", "")) > 60 else ""),
                    "Organism": study.get("organism", ""),
                    "Study Type": study.get("study_type", ""),
                    "Samples": study.get("sample_count", 0),
                    "Sex Metadata %": completeness_pct,
                    "Inferrable": inferrable_status,
                }
            )

        df = pd.DataFrame(display_data)

        st.subheader("Study Results")
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Accession": st.column_config.TextColumn(width="medium"),
                "Title": st.column_config.TextColumn(width="large"),
                "Organism": st.column_config.TextColumn(width="small"),
                "Study Type": st.column_config.TextColumn(width="small"),
                "Samples": st.column_config.NumberColumn(width="small"),
                "Sex Metadata %": st.column_config.TextColumn(width="small"),
                "Inferrable": st.column_config.TextColumn(width="small"),
            },
        )

        # Pagination
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            if st.session_state.page_number > 0:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.page_number -= 1
                    st.rerun()
            else:
                st.write("")

        with col2:
            page_info = f"Page {st.session_state.page_number + 1} ({len(studies)} results shown)"
            st.write(page_info)

        with col3:
            if len(studies) == 50 and (st.session_state.page_number + 1) * 50 < total_results:
                if st.button("Next →", use_container_width=True):
                    st.session_state.page_number += 1
                    st.rerun()
            else:
                st.write("")

except Exception as e:
    st.error(f"Error performing search: {e}")
    st.info("Make sure your Supabase connection is configured correctly.")
