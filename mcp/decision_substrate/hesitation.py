#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bounded Hesitation & Deliberation Budget Controller.

Lifecycle state machine:
- Default initial budget: B_hesitate = 3
- Extension allowance: E_available = True (max 1 human-granted extension)
- Hard convergence lock: Zero hesitation once exhausted
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List, Optional


class HesitationState(str, Enum):
    """Lifecycle states of the bounded deliberation / hesitation budget controller."""
    NORMAL = "normal"                          # budget in [1, 3], extension available
    BUDGET_EXHAUSTED = "budget_exhausted"      # budget == 0, extension available (must apply to user)
    EXTENSION_ACTIVE = "extension_active"      # budget == 1, extension granted and currently active
    HARD_LOCKED = "hard_locked"                # budget == 0, extension exhausted or rejected (hard lock)


class HesitationBudgetManager:
    """Bounded Hesitation Budget Controller (有界犹豫与自主决策预算控制器).
    
    Guarantees:
    1. Default Initial Budget: Exactly 3 hesitation / probe tokens (B_hesitate = 3).
    2. Explicit Human Escalation: Once exhausted (B_hesitate = 0), agent cannot unilaterally continue hesitating.
       Must invoke user escalation (e.g. ask_question) for exactly 1 extension.
    3. Single Extension Cap: Extension can only be applied once (E_available -> False).
    4. Deterministic Hard Convergence: When budget is 0 and extension is unavailable, state is strictly HARD_LOCKED.
       Any further hesitation request is rejected, forcing physical deterministic commit or explicit failure reporting.
    """

    def __init__(self, default_budget: int = 3, session_id: str = "default"):
        self.session_id = session_id
        self.default_budget = default_budget
        self.current_budget = default_budget
        self.extension_available = True
        self.extension_granted = False
        self.extension_consumed = False
        self.history: List[Dict[str, Any]] = []

    def get_state(self) -> HesitationState:
        """Derive the deterministic lifecycle state from budget invariants."""
        if self.current_budget > 0:
            if not self.extension_available:
                return HesitationState.EXTENSION_ACTIVE
            return HesitationState.NORMAL
        else:
            if self.extension_available:
                return HesitationState.BUDGET_EXHAUSTED
            return HesitationState.HARD_LOCKED

    def can_hesitate(self) -> bool:
        """Check if autonomous hesitation is currently permitted."""
        return self.current_budget > 0

    def consume_hesitation(self, reason: str = "") -> Dict[str, Any]:
        """Attempt to consume 1 hesitation allowance."""
        current_state = self.get_state()

        if current_state in (HesitationState.NORMAL, HesitationState.EXTENSION_ACTIVE):
            self.current_budget -= 1
            if current_state == HesitationState.EXTENSION_ACTIVE:
                self.extension_consumed = True

            record = {
                "action": "consume",
                "reason": reason,
                "budget_after": self.current_budget,
                "state_after": self.get_state().value,
                "timestamp": time.time(),
            }
            self.history.append(record)

            return {
                "allowed": True,
                "remaining_budget": self.current_budget,
                "extension_available": self.extension_available,
                "state": self.get_state().value,
                "reason": reason,
                "message": f"Hesitation permitted. Remaining budget: {self.current_budget}."
            }

        elif current_state == HesitationState.BUDGET_EXHAUSTED:
            return {
                "allowed": False,
                "remaining_budget": 0,
                "extension_available": True,
                "state": current_state.value,
                "action_required": "APPLY_TO_HUMAN",
                "message": (
                    "Default hesitation budget (3) exhausted. You must pause autonomous hesitation "
                    "and formally apply to the user for 1 additional hesitation chance."
                ),
            }

        else:  # HARD_LOCKED
            return {
                "allowed": False,
                "remaining_budget": 0,
                "extension_available": False,
                "state": HesitationState.HARD_LOCKED.value,
                "action_required": "HARD_CONVERGENCE",
                "message": (
                    "All hesitation opportunities (3 default + 1 extension) permanently exhausted. "
                    "Zero hesitation permitted. Deterministic commit to execution is strictly enforced."
                ),
            }

    def apply_for_extension(self, user_approved: bool, reason: str = "") -> Dict[str, Any]:
        """Process user's decision on the one-time hesitation extension request."""
        if not self.extension_available:
            return {
                "success": False,
                "error": "EXTENSION_ALREADY_USED_OR_UNAVAILABLE",
                "remaining_budget": self.current_budget,
                "extension_available": False,
                "state": self.get_state().value,
                "message": "The one-time hesitation extension has already been resolved and cannot be reapplied.",
            }

        self.extension_available = False
        if user_approved:
            self.extension_granted = True
            self.current_budget += 1
            record = {
                "action": "extension_granted",
                "reason": reason,
                "budget_after": self.current_budget,
                "state_after": self.get_state().value,
                "timestamp": time.time(),
            }
            self.history.append(record)
            return {
                "success": True,
                "granted": True,
                "remaining_budget": self.current_budget,
                "extension_available": False,
                "state": self.get_state().value,
                "message": "User approved +1 hesitation extension. Exactly 1 final hesitation opportunity is now available.",
            }
        else:
            self.extension_granted = False
            record = {
                "action": "extension_denied",
                "reason": reason,
                "budget_after": self.current_budget,
                "state_after": HesitationState.HARD_LOCKED.value,
                "timestamp": time.time(),
            }
            self.history.append(record)
            return {
                "success": True,
                "granted": False,
                "remaining_budget": self.current_budget,
                "extension_available": False,
                "state": HesitationState.HARD_LOCKED.value,
                "message": "User denied hesitation extension. System permanently locked into deterministic convergence.",
            }

    def get_status(self) -> Dict[str, Any]:
        """Return full telemetry of the hesitation budget controller."""
        state = self.get_state()
        return {
            "session_id": self.session_id,
            "current_budget": self.current_budget,
            "default_budget": self.default_budget,
            "extension_available": self.extension_available,
            "extension_granted": self.extension_granted,
            "extension_consumed": self.extension_consumed,
            "can_hesitate": self.can_hesitate(),
            "state": state.value,
            "history_count": len(self.history),
        }

    def reset(self, default_budget: int = 3):
        """Reset budget controller for a fresh session or testing."""
        self.default_budget = default_budget
        self.current_budget = default_budget
        self.extension_available = True
        self.extension_granted = False
        self.extension_consumed = False
        self.history.clear()
