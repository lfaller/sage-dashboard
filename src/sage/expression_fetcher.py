"""Expression data fetching and parsing from GEO.

Extracts and normalizes gene expression data from GEOparse GSE objects
for sex inference using X and Y chromosome genes.

Based on Flynn et al. (2021) BMC Bioinformatics methodology for sex inference
from gene expression data. Reference: Flynn E, Chang A, Altman RB.
"Large-scale labeling and assessment of sex bias in publicly available
expression data." BMC Bioinformatics. 2021 Mar 30;22(1):168.
doi: 10.1186/s12859-021-04070-2
"""

from typing import Dict, List, Optional
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Sex-specific gene markers for expression-based inference
FEMALE_MARKERS = ["XIST"]
MALE_MARKERS = ["RPS4Y1", "DDX3Y", "EIF1AY", "KDM5D"]
ALL_SEX_MARKERS = FEMALE_MARKERS + MALE_MARKERS


def get_x_chromosome_genes() -> List[str]:
    """Get list of X chromosome genes including escape genes.

    Returns X chromosome genes that are used in sex classification.
    This will be expanded with escape genes from biomaRt in future.
    """
    # Start with female markers
    x_genes = FEMALE_MARKERS.copy()

    # TODO: Add additional X escape genes from biomaRt reference
    # Based on Flynn et al., X escape genes have strongest female signal

    return x_genes


def get_y_chromosome_genes() -> List[str]:
    """Get list of Y chromosome genes.

    Returns Y chromosome genes that are used in sex classification.
    """
    return MALE_MARKERS.copy()


class ExpressionFetcher:
    """Fetches and parses expression data from GEOparse GSE objects."""

    def __init__(self):
        """Initialize ExpressionFetcher."""
        self.x_genes = get_x_chromosome_genes()
        self.y_genes = get_y_chromosome_genes()
        self.all_genes = self.x_genes + self.y_genes

    def fetch_sample_expression(
        self, gse, gsm_id: str, gene_symbols: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Fetch expression values for specific genes from a GSM sample.

        Args:
            gse: GEOparse GSE object
            gsm_id: GSM accession ID (e.g., "GSM123456")
            gene_symbols: List of gene symbols to extract (defaults to sex markers)

        Returns:
            Dict mapping gene symbol to expression value (float)
            Missing genes return 0.0

        Raises:
            ValueError: If expression table format cannot be parsed
            KeyError: If GSM not found in GSE
        """
        if gene_symbols is None:
            gene_symbols = self.all_genes

        try:
            gsm = gse.gsms[gsm_id]
        except KeyError:
            raise KeyError(f"GSM {gsm_id} not found in GSE {gse.name}")

        if not hasattr(gsm, "table") or gsm.table is None or gsm.table.empty:
            raise ValueError(f"No expression table available for {gsm_id}")

        table = gsm.table

        # Find gene identifier column
        gene_col = self._find_gene_column(table)
        if gene_col is None:
            raise ValueError(f"Could not find gene identifier column in {gsm_id}")

        # Find expression value column
        value_col = self._find_value_column(table)
        if value_col is None:
            raise ValueError(f"Could not find expression value column in {gsm_id}")

        # Extract expression for target genes
        result = {}
        for gene in gene_symbols:
            try:
                # Match gene (case-insensitive)
                rows = table[table[gene_col].str.upper() == gene.upper()]
                if not rows.empty:
                    value = float(rows[value_col].iloc[0])
                    result[gene] = value
                else:
                    # Gene not detected
                    result[gene] = 0.0
            except (ValueError, TypeError, AttributeError):
                # Handle conversion errors, default to 0
                result[gene] = 0.0

        return result

    def fetch_study_expression(self, gse, gene_symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch expression matrix for all samples in a study.

        Args:
            gse: GEOparse GSE object
            gene_symbols: List of genes to extract (defaults to sex markers)

        Returns:
            DataFrame with shape (n_samples, n_genes)
            Genes as columns, samples as rows
            Sample names as index

        Note:
            Handles missing genes by filling with 0.0
            Logs warnings for samples with expression fetch errors
        """
        if gene_symbols is None:
            gene_symbols = self.all_genes

        expression_matrix = []
        sample_ids = []
        errors = []

        for gsm_id in gse.gsms.keys():
            try:
                expression = self.fetch_sample_expression(gse, gsm_id, gene_symbols)
                expression_matrix.append(expression)
                sample_ids.append(gsm_id)
            except Exception as e:
                errors.append((gsm_id, str(e)))
                logger.warning(f"Could not fetch expression for {gsm_id}: {e}")

        if errors:
            logger.warning(
                f"Failed to fetch expression for {len(errors)}/{len(gse.gsms)} samples in {gse.name}"
            )

        if not expression_matrix:
            raise ValueError(f"Could not fetch expression data for any samples in {gse.name}")

        # Convert to DataFrame
        df = pd.DataFrame(expression_matrix, index=sample_ids)

        return df

    @staticmethod
    def _find_gene_column(table: pd.DataFrame) -> Optional[str]:
        """Find gene identifier column in expression table.

        Tries common column names in order:
        1. Case-insensitive matches to standard names
        2. First column (fallback)
        """
        standard_names = [
            "GENE_SYMBOL",
            "Gene Symbol",
            "Symbol",
            "gene_name",
            "IDENTIFIER",
            "ID",
        ]

        # Try exact case-insensitive matches
        table_cols_upper = {col.upper(): col for col in table.columns}

        for name in standard_names:
            if name.upper() in table_cols_upper:
                return table_cols_upper[name.upper()]

        # Fallback to first column
        if len(table.columns) > 0:
            return table.columns[0]

        return None

    @staticmethod
    def _find_value_column(table: pd.DataFrame) -> Optional[str]:
        """Find expression value column in expression table.

        Tries common column names in order:
        1. Quantification columns (TPM, FPKM, counts)
        2. "VALUE"
        3. Last non-gene column (fallback)
        """
        standard_names = ["VALUE", "TPM", "FPKM", "counts", "normalized_count", "log2"]

        # Try exact case-insensitive matches
        table_cols_upper = {col.upper(): col for col in table.columns}

        for name in standard_names:
            if name.upper() in table_cols_upper:
                return table_cols_upper[name.upper()]

        # Fallback to last column (usually values)
        if len(table.columns) > 1:
            return table.columns[-1]

        return None
