"""An example page-selection policy, used only in tests (test_policy_registry.py): the plug-in path.

``first_pages_example`` fills the capacity Selector V2 leaves after the mandatory and recent Pages with the oldest
candidate Pages first (the history right after the initial prompt). It is one PagePolicy subclass and one
registration; nothing else in the runtime names it. A test registers it for the duration of a block:

    with policy_registry.registered(FirstPagesExample()):
        ...  # serve.py --policy first_pages_example, its contract pin, plans, receipts and evidence

Its properties are the plug-in defaults (policy_registry.PagePolicy): kind ``window``, Selector V2 with a fixed
budget (``--active-ratio`` / ``--active-token-budget``), no query vector, no Goal engine, its pin
``{order: 'oldest_first_example_v1'}`` under its own name in the selection contract, and its record under
``query.selection.window``.
"""
import policy_registry


class FirstPagesExample(policy_registry.PagePolicy):
    name = 'first_pages_example'
    order_id = 'oldest_first_example_v1'

    def select(self, context):
        # The candidates oldest first; the runtime takes the first capacity - mandatory - recent of them.
        return policy_registry.page_selection(self.name, sorted(context.candidates))
