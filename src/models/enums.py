"""Shared enumerations used across typed models and graph state."""

from __future__ import annotations

from enum import Enum


class GoalStatus(str, Enum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class AgentName(str, Enum):
    GOAL_MANAGER = "goal_manager"
    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    INTAKE = "intake"
    DOCUMENT = "document"
    MEDICAL_EXTRACTION = "medical_extraction"
    MINIMUM_CRITERIA = "minimum_criteria"
    SERIOUSNESS_TRIAGE = "seriousness_triage"
    DUPLICATE_SEARCH = "duplicate_search"
    MISSING_INFORMATION = "missing_information"
    NARRATIVE = "narrative"
    EVALUATOR = "evaluator"
    HUMAN_REVIEW = "human_review"


class DecisionOutcome(str, Enum):
    PROCEED = "proceed"
    RETRY = "retry"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    REJECT = "reject"
    STOP = "stop"


class ConflictStatus(str, Enum):
    NONE = "none"
    CONFLICTING = "conflicting"
    MISSING = "missing"


class FileValidationStatus(str, Enum):
    VALID = "valid"
    UNSUPPORTED_TYPE = "unsupported_type"
    SIZE_EXCEEDED = "size_exceeded"
    PAGE_LIMIT_EXCEEDED = "page_limit_exceeded"
    SIGNATURE_MISMATCH = "signature_mismatch"
    REJECTED = "rejected"


class OCRQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    UNREADABLE = "unreadable"
    NOT_APPLICABLE = "not_applicable"


class SeriousnessIndicator(str, Enum):
    HOSPITALIZATION = "hospitalization"
    LIFE_THREATENING = "life_threatening"
    DEATH = "death"
    DISABILITY = "disability"
    CONGENITAL_ANOMALY = "congenital_anomaly"
    OTHER_MEDICALLY_IMPORTANT = "other_medically_important"


class ReviewDecision(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class RouteReason(str, Enum):
    NORMAL = "normal"
    UNSUPPORTED_FILE = "unsupported_file"
    LOW_OCR_QUALITY = "low_ocr_quality"
    NON_SAFETY_CONTENT = "non_safety_content"
    MISSING_MINIMUM_CRITERIA = "missing_minimum_criteria"
    POTENTIAL_DUPLICATE = "potential_duplicate"
    VALIDATION_FAILURE = "validation_failure"
    LLM_UNAVAILABLE = "llm_unavailable"


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    RETRIED = "retried"


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryOperation(str, Enum):
    READ = "read"
    WRITE = "write"
