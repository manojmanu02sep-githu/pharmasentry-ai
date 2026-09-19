"""Typed Pydantic domain models for PharmaSentry AI.

Import from this package (e.g. ``from src.models import CaseGoal``) rather
than reaching into individual submodules, so the public surface stays stable
as internal files are reorganized.
"""

from src.models.audit import (
    AgentEvent,
    ErrorRecord,
    MemoryEvent,
    RetrievalEvent,
    ToolEvent,
    TraceContext,
)
from src.models.criteria import MinimumCriteriaResult
from src.models.duplicates import DuplicateCandidate
from src.models.enums import (
    AgentName,
    ConflictStatus,
    DecisionOutcome,
    FileValidationStatus,
    GoalStatus,
    MemoryOperation,
    MemoryTier,
    OCRQuality,
    ReviewDecision,
    RouteReason,
    ToolCallStatus,
    TriageIndicator,
    TriagePriority,
)
from src.models.evaluation import EvaluationCheck, EvaluationResult
from src.models.evidence import Citation, FieldValue, SourcePassage
from src.models.extraction import (
    EventInfo,
    ExtractedFields,
    OutcomeInfo,
    PatientInfo,
    ProductInfo,
    ReporterInfo,
    TreatmentInfo,
)
from src.models.goal import CaseGoal, SuccessCriterion
from src.models.intake import Attachment, EmailMessage
from src.models.missing_info import FollowUpDraft, MissingInformationItem
from src.models.plan import ExecutionPlan, PlanRevision, PlanStep
from src.models.reasoning import AgentDecision
from src.models.report import ReportSection, TriageReport
from src.models.review import ReviewerChange, ReviewStatus, ReviewTask
from src.models.triage import TriageFinding, TriageResult

__all__ = [
    "AgentDecision",
    "AgentEvent",
    "AgentName",
    "Attachment",
    "CaseGoal",
    "Citation",
    "ConflictStatus",
    "DecisionOutcome",
    "DuplicateCandidate",
    "EmailMessage",
    "ErrorRecord",
    "EvaluationCheck",
    "EvaluationResult",
    "EventInfo",
    "ExecutionPlan",
    "ExtractedFields",
    "FieldValue",
    "FileValidationStatus",
    "FollowUpDraft",
    "GoalStatus",
    "MemoryEvent",
    "MemoryOperation",
    "MemoryTier",
    "MinimumCriteriaResult",
    "MissingInformationItem",
    "OCRQuality",
    "OutcomeInfo",
    "PatientInfo",
    "PlanRevision",
    "PlanStep",
    "ProductInfo",
    "ReportSection",
    "ReporterInfo",
    "RetrievalEvent",
    "ReviewDecision",
    "ReviewStatus",
    "ReviewTask",
    "ReviewerChange",
    "RouteReason",
    "SourcePassage",
    "SuccessCriterion",
    "ToolCallStatus",
    "ToolEvent",
    "TraceContext",
    "TreatmentInfo",
    "TriageFinding",
    "TriageIndicator",
    "TriagePriority",
    "TriageReport",
    "TriageResult",
]
