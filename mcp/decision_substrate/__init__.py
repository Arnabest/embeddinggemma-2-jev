#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified Decision Substrate Package.

Top-level exports:
- DecisionSubstrate: Master substrate controller
- BaseDecisionModelAdapter: Abstract adapter interface
- HeuristicDecisionAdapter, JevGemmaDecisionAdapter, CompositeDecisionAdapter
- IntentDecision, HazardDecision, TriageDecision, DivergenceDecision, CompletionDecision
- PreTurnGrounding, PostTurnAudit
- EpistemicArbiter, EpistemicContext
- HesitationBudgetManager, HesitationState
"""

from .protocol import (
    BaseDecisionModelAdapter,
    CompletionDecision,
    DivergenceDecision,
    HazardDecision,
    IntentDecision,
    IntentType,
    PostTurnAudit,
    PreTurnGrounding,
    TriageDecision,
)
from .epistemic import (
    AuthoritativeDomainEngine,
    EpistemicArbiter,
    EpistemicContext,
    EpistemicWeightMatrix,
)
from .hesitation import (
    HesitationBudgetManager,
    HesitationState,
)
from .adapters import (
    CompositeDecisionAdapter,
    HeuristicDecisionAdapter,
    JevGemmaDecisionAdapter,
)
from .kernel import DecisionSubstrate

__all__ = [
    "DecisionSubstrate",
    "BaseDecisionModelAdapter",
    "HeuristicDecisionAdapter",
    "JevGemmaDecisionAdapter",
    "CompositeDecisionAdapter",
    "IntentDecision",
    "HazardDecision",
    "TriageDecision",
    "DivergenceDecision",
    "CompletionDecision",
    "PreTurnGrounding",
    "PostTurnAudit",
    "IntentType",
    "EpistemicArbiter",
    "EpistemicContext",
    "EpistemicWeightMatrix",
    "AuthoritativeDomainEngine",
    "HesitationBudgetManager",
    "HesitationState",
]
