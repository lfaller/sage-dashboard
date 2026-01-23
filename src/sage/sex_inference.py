"""Sex inference module for identifying rescue opportunities.

Uses strategy pattern for extensibility. MVP implements metadata-based
inference; architecture supports future gene expression analysis.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import re


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class InferenceResult:
    """Result of sex inference analysis."""

    inferrable: bool
    confidence: float  # 0.0-1.0
    method: str  # "metadata", "expression", etc.
    factors: Dict


# ============================================================================
# Strategy Pattern (for extensibility)
# ============================================================================


class InferenceStrategy(ABC):
    """Abstract base class for inference strategies."""

    @abstractmethod
    def infer(self, study_dict: Dict) -> InferenceResult:
        """Perform inference on a study."""
        pass


class MetadataInferenceStrategy(InferenceStrategy):
    """Infer sex from study metadata and sample naming patterns (MVP)."""

    def infer(self, study_dict: Dict) -> InferenceResult:
        """Main inference logic for metadata strategy."""
        factors = self._analyze_factors(study_dict)
        confidence = self._calculate_confidence(factors)
        inferrable = bool(confidence >= 0.5)  # Threshold for "inferrable"

        return InferenceResult(
            inferrable=inferrable, confidence=confidence, method="metadata", factors=factors
        )

    def _analyze_factors(self, study_dict: Dict) -> Dict:
        """Analyze all factors that contribute to inference."""
        study_type = study_dict.get("study_type")
        sample_count = study_dict.get("sample_count", 0)
        organism = study_dict.get("organism", "")
        has_sex_metadata = study_dict.get("has_sex_metadata", False)

        # Analyze sample names if available
        sample_names = study_dict.get("sample_names", [])
        sample_analysis = analyze_sample_names(sample_names)

        return {
            "is_rna_seq": study_type == "RNA-seq",
            "sample_count": sample_count,
            "has_sufficient_samples": sample_count >= 20,
            "is_human": organism == "Homo sapiens",
            "sample_name_confidence": sample_analysis["confidence"],
            "sample_name_pattern": sample_analysis["pattern"],
            "has_sex_metadata": has_sex_metadata,
        }

    def _calculate_confidence(self, factors: Dict) -> float:
        """Calculate confidence score from factors (weighted formula)."""
        score = 0.0

        # RNA-seq is ideal for expression-based inference
        if factors["is_rna_seq"]:
            score += 0.35

        # Need decent sample size
        if factors["has_sufficient_samples"]:
            score += 0.25

        # Human genomics better characterized
        if factors["is_human"]:
            score += 0.15

        # Sample name pattern bonus (up to 0.25)
        sample_confidence = factors.get("sample_name_confidence", 0.0)
        score += sample_confidence * 0.25

        # Normalize to [0.0, 1.0]
        return min(1.0, max(0.0, score))


# ============================================================================
# Helper Functions
# ============================================================================


def analyze_sample_names(sample_names: List[str]) -> Dict:
    """Analyze sample names for sex pattern indicators.

    Looks for patterns like M/F, Male/Female (case insensitive).

    Args:
        sample_names: List of sample name strings

    Returns:
        Dict with:
            - pattern: "clear", "partial", or "none"
            - confidence: 0.0-1.0
            - male_count: Number of male-labeled samples
            - female_count: Number of female-labeled samples
    """
    if not sample_names:
        return {"pattern": "none", "confidence": 0.0, "male_count": 0, "female_count": 0}

    # Regex patterns for sex indicators
    # Match: M/F followed by digit/dash/dot, OR full "male"/"female" surrounded by non-alphas
    male_pattern = re.compile(r"(m[0-9_\-\.]+|(?:^|[^a-z])male(?:[^a-z]|$))", re.IGNORECASE)
    female_pattern = re.compile(r"(f[0-9_\-\.]+|(?:^|[^a-z])female(?:[^a-z]|$))", re.IGNORECASE)

    male_count = 0
    female_count = 0

    for name in sample_names:
        if name is None:
            continue
        name_str = str(name)

        if male_pattern.search(name_str):
            male_count += 1
        elif female_pattern.search(name_str):
            female_count += 1

    total = len([n for n in sample_names if n is not None])
    labeled = male_count + female_count

    if total == 0:
        confidence = 0.0
        pattern = "none"
    elif labeled == total:
        confidence = 1.0
        pattern = "clear"
    elif labeled >= total * 0.5:
        confidence = labeled / total
        pattern = "partial"
    else:
        confidence = 0.0
        pattern = "none"

    return {
        "pattern": pattern,
        "confidence": confidence,
        "male_count": male_count,
        "female_count": female_count,
    }


def extract_sex_from_characteristics(characteristics: List[str]) -> Optional[str]:
    """Extract sex/gender from characteristics_ch1 list.

    Parses characteristics for sex/gender metadata with support for various
    encoding formats (key-value pairs with different delimiters and key names).

    Supported formats:
        - Standard: "sex: male", "Sex: Female"
        - Abbreviated values: "sex: M", "gender: F"
        - Alternate delimiters: "sex=male", "sex|female"
        - Alternate keys: "gender:", "sample_sex:", "sex_ch1:"

    Args:
        characteristics: List of characteristics strings (e.g., ["sex: male", "age: 45"])

    Returns:
        "male", "female", or None if not found or ambiguous (conflicting values)
    """
    if not characteristics:
        return None

    # Pattern to match various key names and delimiters
    # Keys: sex, gender, sample_sex, sex_ch1 (case-insensitive)
    # Delimiters: :, =, | with optional whitespace
    sex_pattern = re.compile(r"^\s*(sex|gender|sample_sex|sex_ch1)\s*[:=|]\s*(.+?)$", re.IGNORECASE)

    found_values = set()

    for char in characteristics:
        if not char or not isinstance(char, str):
            continue

        match = sex_pattern.match(char)
        if not match:
            continue

        key, value = match.groups()
        value = value.strip().lower()

        # Handle standard full text values
        if value in {"male", "man"}:
            found_values.add("male")
        elif value in {"female", "woman"}:
            found_values.add("female")
        # Handle abbreviated values (single letter, case-insensitive)
        elif value == "m":
            found_values.add("male")
        elif value == "f":
            found_values.add("female")
        # Other formats not recognized - skip

    # Return None if conflicting values found
    if len(found_values) > 1:
        return None

    # Return the found value or None
    return found_values.pop() if found_values else None


def analyze_sample_characteristics(samples_characteristics: List[List[str]]) -> Dict:
    """Analyze sex metadata from sample characteristics across multiple samples.

    Parses characteristics_ch1 for all samples and determines the pattern
    and confidence of sex metadata detection.

    Args:
        samples_characteristics: List of characteristics_ch1 lists (one per sample)

    Returns:
        Dict with:
            - pattern: "clear" (100%), "partial" (50-99%), or "none" (<50%)
            - confidence: 0.0-1.0 (fraction of samples with sex metadata)
            - male_count: Number of male samples
            - female_count: Number of female samples
            - source: "characteristics"
    """
    if not samples_characteristics:
        return {
            "pattern": "none",
            "confidence": 0.0,
            "male_count": 0,
            "female_count": 0,
            "source": "characteristics",
        }

    male_count = 0
    female_count = 0

    for sample_chars in samples_characteristics:
        if not isinstance(sample_chars, list):
            continue

        sex = extract_sex_from_characteristics(sample_chars)
        if sex == "male":
            male_count += 1
        elif sex == "female":
            female_count += 1

    total = len(samples_characteristics)
    labeled = male_count + female_count

    if labeled == total:
        confidence = 1.0
        pattern = "clear"
    elif labeled >= total * 0.5:
        confidence = labeled / total
        pattern = "partial"
    else:
        confidence = 0.0
        pattern = "none"

    return {
        "pattern": pattern,
        "confidence": confidence,
        "male_count": male_count,
        "female_count": female_count,
        "source": "characteristics",
    }


def merge_sex_analyses(characteristics_result: Dict, sample_names_result: Dict) -> Dict:
    """Merge sex detection results from characteristics and sample names.

    Prioritizes characteristics over sample names as they are more explicit
    and reliable. Returns result from the source with highest confidence.

    Args:
        characteristics_result: Dict from analyze_sample_characteristics()
        sample_names_result: Dict from analyze_sample_names()

    Returns:
        Merged dict with pattern, confidence, counts, and source tracking
    """
    chars_confidence = characteristics_result.get("confidence", 0.0)
    names_confidence = sample_names_result.get("confidence", 0.0)

    # Priority logic:
    # 1. If characteristics has confidence >= 0.5, use it
    # 2. Else if sample names has confidence >= 0.5, use it
    # 3. Else use whichever has higher confidence
    if chars_confidence >= 0.5:
        result = characteristics_result.copy()
        result["source"] = "characteristics"
    elif names_confidence >= 0.5:
        result = sample_names_result.copy()
        result["source"] = "sample_names"
    elif chars_confidence > names_confidence:
        result = characteristics_result.copy()
        result["source"] = "characteristics"
    else:
        result = sample_names_result.copy()
        result["source"] = "sample_names"

    return result


def calculate_confidence(factors: Dict) -> float:
    """Calculate confidence score from factors (weighted formula).

    Weights:
        - RNA-seq: 35%
        - Sample size (≥20): 25%
        - Sample name confidence: 25%
        - Human organism: 15%

    Args:
        factors: Dict with inference factors

    Returns:
        Confidence score in [0.0, 1.0]
    """
    score = 0.0

    if factors.get("is_rna_seq", False):
        score += 0.35

    if factors.get("has_sufficient_samples", False):
        score += 0.25

    if factors.get("is_human", False):
        score += 0.15

    sample_confidence = factors.get("sample_name_confidence", 0.0)
    score += sample_confidence * 0.25

    return min(1.0, max(0.0, score))


def infer_from_metadata(study_dict: Dict) -> Dict:
    """Main public API for metadata-based inference.

    Args:
        study_dict: Study record dict

    Returns:
        Dict compatible with database update with keys:
            - sex_inferrable: bool
            - sex_inference_confidence: float (0.0-1.0)
            - inference_method: str
            - inference_factors: Dict
    """
    strategy = MetadataInferenceStrategy()
    result = strategy.infer(study_dict)

    return {
        "sex_inferrable": result.inferrable,
        "sex_inference_confidence": result.confidence,
        "inference_method": result.method,
        "inference_factors": result.factors,
    }


# ============================================================================
# Expression-Based Inference (Phase 6A)
# ============================================================================


class SexClassifier:
    """Classifies sex based on gene expression patterns using elastic net model.

    Based on Flynn et al. (2021) - uses penalized logistic regression on
    X and Y chromosome genes for robust, platform-agnostic sex classification.
    """

    def __init__(
        self, x_genes: Optional[List[str]] = None, y_genes: Optional[List[str]] = None, model=None
    ):
        """Initialize SexClassifier.

        Args:
            x_genes: List of X chromosome gene symbols
            y_genes: List of Y chromosome gene symbols
            model: Pre-trained LogisticRegression model (optional)
        """
        from src.sage.expression_fetcher import get_x_chromosome_genes, get_y_chromosome_genes

        self.x_genes = x_genes or get_x_chromosome_genes()
        self.y_genes = y_genes or get_y_chromosome_genes()
        self.all_genes = self.x_genes + self.y_genes
        self.model = model
        self.model_version = "elasticnet_v1"

    def train(
        self, expression_matrix, sex_labels, alpha: float = 0.5, l1_ratio: float = 0.5
    ) -> None:
        """Train elastic net logistic regression model.

        Args:
            expression_matrix: Shape (n_samples, n_genes) with normalized expression
            sex_labels: Array of 0 (female) or 1 (male)
            alpha: Regularization strength
            l1_ratio: Balance between L1 (lasso) and L2 (ridge)
        """
        from sklearn.linear_model import LogisticRegression

        self.model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=1.0 / alpha,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=42,
        )
        self.model.fit(expression_matrix, sex_labels)

    def predict_sex_score(self, sample_expression: Dict) -> float:
        """Get probability score that sample is male [0-1].

        Args:
            sample_expression: Dict mapping gene symbol to expression value

        Returns:
            float: P(male), where 0 = female, 1 = male

        Raises:
            ValueError: If model not trained
        """
        import numpy as np

        if self.model is None:
            raise ValueError("Model not trained")

        # Convert to proper format
        expression_array = np.array(
            [sample_expression.get(g, 0.0) for g in self.all_genes]
        ).reshape(1, -1)
        prob_male = self.model.predict_proba(expression_array)[0, 1]
        return float(prob_male)

    def classify_sample(self, sample_expression: Dict, threshold: float = 0.7) -> tuple:
        """Classify single sample as male/female/ambiguous.

        Args:
            sample_expression: Expression dict mapping gene symbol to value
            threshold: Classification threshold (Flynn et al. used 0.7)

        Returns:
            Tuple of (sex: str, confidence: float)
                - sex: "male", "female", or "ambiguous"
                - confidence: 0.0-1.0
        """
        prob_male = self.predict_sex_score(sample_expression)

        if prob_male >= threshold:
            return "male", prob_male
        elif prob_male <= (1 - threshold):
            return "female", 1 - prob_male
        else:
            return "ambiguous", 0.5 - abs(prob_male - 0.5)

    def classify_study(self, expression_df, threshold: float = 0.7):
        """Classify all samples in a study.

        Args:
            expression_df: DataFrame with genes as columns, samples as rows
            threshold: Classification threshold

        Returns:
            DataFrame with columns: sample_id, inferred_sex, confidence
        """
        import pandas as pd

        results = []

        for sample_id, row in expression_df.iterrows():
            sex, confidence = self.classify_sample(row.to_dict(), threshold)
            results.append({"sample_id": sample_id, "inferred_sex": sex, "confidence": confidence})

        return pd.DataFrame(results)


class ElasticNetInferenceStrategy(InferenceStrategy):
    """Infer sex from gene expression patterns using elastic net model.

    Based on Flynn et al. (2021) BMC Bioinformatics paper.
    Analyzes X and Y chromosome gene expression to classify biological sex.
    """

    def __init__(self, classifier: Optional[SexClassifier] = None):
        """Initialize ElasticNetInferenceStrategy.

        Args:
            classifier: Pre-trained SexClassifier (optional)
        """
        self.classifier = classifier or SexClassifier()

    def infer(self, study_dict: Dict) -> InferenceResult:
        """Perform expression-based inference.

        Args:
            study_dict: Study dict that must contain 'gse' key with GEOparse GSE object

        Returns:
            InferenceResult with sample-level classifications

        Raises:
            ValueError: If study_dict missing 'gse' key or expression not available
        """
        import logging
        from src.sage.expression_fetcher import ExpressionFetcher

        logger = logging.getLogger(__name__)

        gse = study_dict.get("gse")
        if not gse:
            raise ValueError("study_dict must contain 'gse' key with GEOparse GSE object")

        fetcher = ExpressionFetcher()

        try:
            # Fetch expression for sex markers
            expression_df = fetcher.fetch_study_expression(gse)
        except Exception as e:
            logger.warning(f"Could not fetch expression for {gse.name}: {e}")
            return InferenceResult(
                inferrable=False,
                confidence=0.0,
                method="expression_elasticnet",
                factors={"error": str(e), "gse": gse.name},
            )

        # Classify samples
        classifications = self.classifier.classify_study(expression_df)

        # Calculate study-level statistics
        avg_confidence = classifications["confidence"].mean()
        inferrable = bool(avg_confidence >= 0.7)

        return InferenceResult(
            inferrable=inferrable,
            confidence=float(avg_confidence),
            method="expression_elasticnet",
            factors={
                "sample_count": len(classifications),
                "female_count": int((classifications["inferred_sex"] == "female").sum()),
                "male_count": int((classifications["inferred_sex"] == "male").sum()),
                "ambiguous_count": int((classifications["inferred_sex"] == "ambiguous").sum()),
                "gse": gse.name,
                "classifications": classifications.to_dict("records"),
            },
        )
