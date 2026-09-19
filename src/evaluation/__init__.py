"""Golden dataset schemas, loaders, and the baseline metrics framework.

The Evaluator Agent (Phase 6) and full trajectory evaluation build on top
of this package; nothing here calls an LLM or an agent.
"""

from src.evaluation.golden_loader import (
    GoldenDatasetError,
    cases_by_family,
    dataset_as_json_dict,
    load_all_cases,
    load_case,
    load_cases,
    load_manifest,
    load_split,
    load_test_cases,
    load_train_cases,
    load_validation_cases,
    validate_no_family_leakage,
)
from src.evaluation.metrics import (
    PrecisionRecallF1,
    SensitivitySpecificity,
    citation_coverage,
    classification_prf1,
    duplicate_precision_at_k,
    duplicate_recall_at_k,
    extraction_prf1,
    mean_duplicate_precision_at_k,
    mean_duplicate_recall_at_k,
    mean_reciprocal_rank,
    triage_escalation_sensitivity_specificity,
    unsupported_claim_rate,
)
from src.evaluation.schemas import (
    FIELD_KEYS,
    AttachmentMetadata,
    DatasetManifest,
    DatasetSplit,
    ExpectedMinimumCriteria,
    GoldenCase,
    GoldenCaseCategory,
    SafetyClassification,
)

__all__ = [
    "FIELD_KEYS",
    "AttachmentMetadata",
    "DatasetManifest",
    "DatasetSplit",
    "ExpectedMinimumCriteria",
    "GoldenCase",
    "GoldenCaseCategory",
    "GoldenDatasetError",
    "PrecisionRecallF1",
    "SafetyClassification",
    "SensitivitySpecificity",
    "cases_by_family",
    "citation_coverage",
    "classification_prf1",
    "dataset_as_json_dict",
    "duplicate_precision_at_k",
    "duplicate_recall_at_k",
    "extraction_prf1",
    "load_all_cases",
    "load_case",
    "load_cases",
    "load_manifest",
    "load_split",
    "load_test_cases",
    "load_train_cases",
    "load_validation_cases",
    "mean_duplicate_precision_at_k",
    "mean_duplicate_recall_at_k",
    "mean_reciprocal_rank",
    "triage_escalation_sensitivity_specificity",
    "unsupported_claim_rate",
    "validate_no_family_leakage",
]
