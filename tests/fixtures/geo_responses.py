"""Mock GEO responses for testing geo_fetcher module."""

from unittest.mock import Mock


# ============================================================================
# RNA-seq Study with Sex Metadata in Sample Names
# ============================================================================

MOCK_GSE_RNA_SEQ_WITH_SEX = Mock()
MOCK_GSE_RNA_SEQ_WITH_SEX.name = "GSE123456"
MOCK_GSE_RNA_SEQ_WITH_SEX.metadata = {
    "title": ["RNA-seq of breast cancer samples with sex differences"],
    "organism": ["Homo sapiens"],
    "type": ["Expression profiling by high throughput sequencing"],
    "summary": ["This study investigates sex-specific expression patterns in breast cancer."],
    "submission_date": ["2023-05-15"],
    "platform_id": ["GPL24676"],
    "pubmed_id": ["12345678"],
}
MOCK_GSE_RNA_SEQ_WITH_SEX.gsms = {
    "GSM1": Mock(metadata={"title": ["Male_sample_1"]}),
    "GSM2": Mock(metadata={"title": ["Female_sample_1"]}),
    "GSM3": Mock(metadata={"title": ["Male_sample_2"]}),
    "GSM4": Mock(metadata={"title": ["Female_sample_2"]}),
    "GSM5": Mock(metadata={"title": ["Male_sample_3"]}),
    "GSM6": Mock(metadata={"title": ["Female_sample_3"]}),
}


# ============================================================================
# RNA-seq Study without Sex Metadata
# ============================================================================

MOCK_GSE_RNA_SEQ_NO_SEX = Mock()
MOCK_GSE_RNA_SEQ_NO_SEX.name = "GSE234567"
MOCK_GSE_RNA_SEQ_NO_SEX.metadata = {
    "title": ["Lung cancer gene expression study"],
    "organism": ["Homo sapiens"],
    "type": ["Expression profiling by high throughput sequencing"],
    "summary": ["RNA-seq analysis of lung cancer patients"],
    "submission_date": ["2022-03-10"],
    "platform_id": ["GPL20795"],
}
MOCK_GSE_RNA_SEQ_NO_SEX.gsms = {
    "GSM7": Mock(metadata={"title": ["Sample_001"]}),
    "GSM8": Mock(metadata={"title": ["Sample_002"]}),
    "GSM9": Mock(metadata={"title": ["Sample_003"]}),
    "GSM10": Mock(metadata={"title": ["Sample_004"]}),
}


# ============================================================================
# Microarray Study with Sex Metadata
# ============================================================================

MOCK_GSE_MICROARRAY_WITH_SEX = Mock()
MOCK_GSE_MICROARRAY_WITH_SEX.name = "GSE789012"
MOCK_GSE_MICROARRAY_WITH_SEX.metadata = {
    "title": ["Gene expression in liver disease by sex"],
    "organism": ["Homo sapiens"],
    "type": ["Expression profiling by array"],
    "summary": ["Microarray analysis of liver samples from male and female patients"],
    "submission_date": ["2022-07-20"],
    "platform_id": ["GPL570"],
}
MOCK_GSE_MICROARRAY_WITH_SEX.gsms = {
    "GSM11": Mock(metadata={"title": ["M01"]}),
    "GSM12": Mock(metadata={"title": ["F01"]}),
    "GSM13": Mock(metadata={"title": ["M02"]}),
    "GSM14": Mock(metadata={"title": ["F02"]}),
}


# ============================================================================
# Microarray Study without Sex Metadata
# ============================================================================

MOCK_GSE_MICROARRAY_NO_SEX = Mock()
MOCK_GSE_MICROARRAY_NO_SEX.name = "GSE345678"
MOCK_GSE_MICROARRAY_NO_SEX.metadata = {
    "title": ["Diabetes risk gene expression"],
    "organism": ["Homo sapiens"],
    "type": ["Expression profiling by array"],
    "summary": ["Gene expression analysis in diabetes patients"],
    "submission_date": ["2021-11-05"],
    "platform_id": ["GPL96"],
}
MOCK_GSE_MICROARRAY_NO_SEX.gsms = {
    "GSM15": Mock(metadata={"title": ["Diabetes_patient_1"]}),
    "GSM16": Mock(metadata={"title": ["Diabetes_patient_2"]}),
    "GSM17": Mock(metadata={"title": ["Control_1"]}),
}


# ============================================================================
# Non-Human Study (Mouse)
# ============================================================================

MOCK_GSE_MOUSE = Mock()
MOCK_GSE_MOUSE.name = "GSE456789"
MOCK_GSE_MOUSE.metadata = {
    "title": ["Mouse immune response study"],
    "organism": ["Mus musculus"],
    "type": ["Expression profiling by high throughput sequencing"],
    "submission_date": ["2023-01-15"],
    "platform_id": ["GPL21103"],
}
MOCK_GSE_MOUSE.gsms = {
    "GSM18": Mock(metadata={"title": ["Male_mouse_1"]}),
    "GSM19": Mock(metadata={"title": ["Female_mouse_1"]}),
}


# ============================================================================
# Study with Missing Optional Fields
# ============================================================================

MOCK_GSE_MINIMAL = Mock()
MOCK_GSE_MINIMAL.name = "GSE567890"
MOCK_GSE_MINIMAL.metadata = {
    "title": ["Simple study"],
    "organism": ["Homo sapiens"],
    # Note: type, summary, platform_id intentionally missing
}
MOCK_GSE_MINIMAL.gsms = {
    "GSM20": Mock(metadata={"title": ["Sample1"]}),
    "GSM21": Mock(metadata={"title": ["Sample2"]}),
}


# ============================================================================
# Study with Empty Sample List
# ============================================================================

MOCK_GSE_EMPTY_SAMPLES = Mock()
MOCK_GSE_EMPTY_SAMPLES.name = "GSE678901"
MOCK_GSE_EMPTY_SAMPLES.metadata = {
    "title": ["Study with no samples yet"],
    "organism": ["Homo sapiens"],
    "type": ["Expression profiling by high throughput sequencing"],
}
MOCK_GSE_EMPTY_SAMPLES.gsms = {}
