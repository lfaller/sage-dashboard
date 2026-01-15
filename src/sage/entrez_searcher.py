"""Entrez API search for GEO studies.

Uses NCBI E-utilities to query the GEO DataSets database for recent studies
matching specific organism and study type criteria.
"""

from typing import List
from datetime import datetime, timedelta

from Bio import Entrez

from sage.geo_fetcher import RateLimiter
from sage.logging_config import get_logger

logger = get_logger(__name__)


class EntrezSearcher:
    """Searches NCBI GEO via Entrez API for studies matching criteria."""

    # Mapping of organism names to Entrez search terms
    ORGANISM_TERMS = {
        "Homo sapiens": '"Homo sapiens"[Organism]',
        "Mus musculus": '"Mus musculus"[Organism]',
    }

    # Mapping of study types to GEO DataSet Type terms
    STUDY_TYPE_TERMS = {
        "RNA-seq": '"Expression profiling by high throughput sequencing"[DataSet Type]',
        "microarray": '"Expression profiling by array"[DataSet Type]',
    }

    def __init__(self, email: str, rate_limit: float = 2.0):
        """Initialize Entrez searcher.

        Args:
            email: Email address (required by NCBI)
            rate_limit: Requests per second (default: 2.0)
        """
        Entrez.email = email
        self.rate_limiter = RateLimiter(rate_limit)
        logger.info(f"EntrezSearcher initialized (email: {email}, rate: {rate_limit} req/sec)")

    def search_recent_studies(
        self,
        organism: str = "Homo sapiens",
        study_type: str = "RNA-seq",
        years_back: int = 5,
        max_results: int = 500,
    ) -> List[str]:
        """Search for recent GEO studies matching criteria.

        Args:
            organism: Organism name (must be in ORGANISM_TERMS)
            study_type: Study type (must be in STUDY_TYPE_TERMS)
            years_back: How many years back to search (default: 5)
            max_results: Maximum number of results (default: 500)

        Returns:
            List of GEO accession IDs (e.g., ["GSE123456", "GSE789012"])
        """
        # Build date range (rolling window)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        date_range = (
            f'("{start_date.strftime("%Y/%m/%d")}"[PDAT] : '
            f'"{end_date.strftime("%Y/%m/%d")}"[PDAT])'
        )

        # Build search query
        organism_term = self.ORGANISM_TERMS.get(organism)
        study_type_term = self.STUDY_TYPE_TERMS.get(study_type)

        if not organism_term or not study_type_term:
            raise ValueError("Invalid organism or study_type")

        query = f"({organism_term}) AND ({study_type_term}) AND {date_range}"

        logger.info(f"Searching GEO: organism={organism}, type={study_type}, years={years_back}")
        logger.debug(f"Entrez query: {query}")

        try:
            # Respect rate limit
            self.rate_limiter.wait()

            # Execute search
            search_handle = Entrez.esearch(
                db="gds",  # GEO DataSets database
                term=query,
                retmax=max_results,
                usehistory="y",
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            # Extract accession IDs
            id_list = search_results.get("IdList", [])

            if not id_list:
                logger.warning(f"No results found for query: {query}")
                return []

            logger.info(f"Found {len(id_list)} studies")

            # Convert GDS IDs to GSE accessions
            accessions = self._convert_gds_to_gse(id_list)

            logger.info(f"Converted to {len(accessions)} GSE accessions")
            return accessions

        except Exception as e:
            logger.error(f"Entrez search failed: {e}")
            raise

    def _convert_gds_to_gse(self, gds_ids: List[str]) -> List[str]:
        """Convert GDS IDs to GSE accessions.

        Entrez.esearch returns GDS IDs (internal NCBI IDs), but we need
        GSE accessions (e.g., GSE123456) for GEOparse.

        Args:
            gds_ids: List of GDS IDs from Entrez search

        Returns:
            List of GSE accessions
        """
        accessions = []

        for gds_id in gds_ids:
            try:
                self.rate_limiter.wait()

                # Fetch summary for this GDS ID
                summary_handle = Entrez.esummary(db="gds", id=gds_id)
                summary = Entrez.read(summary_handle)
                summary_handle.close()

                # Extract GSE accession from summary
                if summary and len(summary) > 0:
                    accession = summary[0].get("Accession", "")
                    if accession.startswith("GSE"):
                        accessions.append(accession)
                    else:
                        logger.debug(f"Skipping non-GSE accession: {accession}")

            except Exception as e:
                logger.warning(f"Failed to convert GDS ID {gds_id}: {e}")
                continue

        return accessions
