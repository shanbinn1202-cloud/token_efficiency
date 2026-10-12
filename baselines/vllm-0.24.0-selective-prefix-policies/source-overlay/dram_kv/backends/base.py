"""Runtime Backend interface."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..schema import PageCatalog, SelectionPlan, TurnContext


class RuntimeBackend(Protocol):
    name: str

    def execute(
        self, context: TurnContext, catalog: PageCatalog, plan: SelectionPlan
    ) -> Mapping[str, Any]:
        ...

