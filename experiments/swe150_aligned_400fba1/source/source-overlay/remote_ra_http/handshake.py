"""Strict PREPARED_LIVE schema, distinct from legacy capability-only PREPARED."""
from private_authority.protocol import hex64, require

def validate_readiness(report):
    expected = dict(default_enabled=False, code_ready=True, native_ready=True,
        worker_attached=True, cuda_qualified=False, growing_smoke_qualified=False,
        live_multi_turn_supported=False, system_valid=False)
    require(type(report) is dict and set(report) == set(expected) | {'missing', 'worker_receipt_sha256'},
            'PREPARED_LIVE readiness schema mismatch')
    require(all(report[key] is value for key, value in expected.items()),
            'PREPARED_LIVE may attest attachment only, not qualification')
    require(type(report['missing']) is list and report['missing'] == [] and
            hex64(report['worker_receipt_sha256']), 'PREPARED_LIVE worker receipt digest invalid')
    return True
