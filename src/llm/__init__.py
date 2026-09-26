"""LLM provider abstraction: a typed protocol plus a deterministic mock.

CLAUDE.md's LLM capability requires "a configured cloud provider and a
deterministic mock provider for tests" with "Pydantic structured output."
Only classification, drafting, and quality-review synthesis use the LLM
here — everything else (access control, file checks, exact rules, schema
validation, thresholds, routing) stays deterministic code per that same
section.
"""
