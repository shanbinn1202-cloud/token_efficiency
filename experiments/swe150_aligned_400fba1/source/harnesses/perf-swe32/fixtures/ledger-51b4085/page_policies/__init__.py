"""Plug-in page-selection policies (policy-registry-1112): one module per policy, one PagePolicy subclass each.

A policy here is registered by one 'module:Class' entry in policy_registry.PLUGINS (for example
'page_policies.oldest_first:OldestFirst'); nothing else in the runtime names it. The README's section "How to add
a page-selection policy" and docs/policy-registry.md describe the class; tests/policy_example.py is a worked
example (registered by its tests only). Empty: the eight policies of 4bf2c40 are policy_registry's own.
"""
