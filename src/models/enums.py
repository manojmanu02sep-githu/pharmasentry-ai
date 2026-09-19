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
    """Nodes in the Diabetes Injection Safety Email Triage workflow.

    ``SUPERVISOR`` is retained from the prior architecture as the
    cross-cutting delegation/limits controller between Planner and the
    reader agents; it is not one of the 11 pipeline stages the user-facing
    workflow names, but it still governs handoffs between them (see
    CLAUDE.md "Orchestration" and "Context Isolation and Delegation").
    """

    GOAL_AGENT = "goal_agent"
    PLANNER = "planner"
    SUPERVISOR = "supervisor"
    SUBJECT_READER = "subject_reader"
    EMAIL_BODY_READER = "email_body_reader"
    ATTACHMENT_READER = "pdf_docx_reader"
    MEDICAL_EXTRACTION = "medical_extraction"
    TRIAGE = "triage"
    DUPLICATE_SEARCH = "duplicate_search"
    REPORT_GENERATOR = "report_generator"
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


class TriageIndicator(str, Enum):
    """Explicit, keyword-matched indicators. Detection is deterministic and
    NEVER infers severity on its own authority — an indicator match only
    feeds the deterministic suggested-priority mapping in
    ``src.tools.domain.suggest_triage_priority``, and every suggestion still
    requires human confirmation (see TriagePriority)."""

    HOSPITALIZATION = "hospitalization"
    LIFE_THREATENING = "life_threatening"
    DEATH = "death"
    DISABILITY = "disability"
    CONGENITAL_ANOMALY = "congenital_anomaly"
    OTHER_MEDICALLY_IMPORTANT = "other_medically_important"
    SEVERE_HYPOGLYCEMIA = "severe_hypoglycemia"
    SEVERE_INJECTION_SITE_REACTION = "severe_injection_site_reaction"


class TriagePriority(str, Enum):
    """AI-suggested triage priority ONLY. Never a final clinical or
    regulatory determination — human review is mandatory before any action
    is taken on a case (see CLAUDE.md Absolute Boundaries)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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
