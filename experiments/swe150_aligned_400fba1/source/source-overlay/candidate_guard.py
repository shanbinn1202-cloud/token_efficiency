"""Hard-disabled review artifact. No setting grants installation or serving."""
import os

STATUS = "NOT_INSTALLABLE"
SERVING_AUTHORIZED = False
GAPS = {
    "incomplete_package": "Partial copied source; full installed package/native libraries/imports are unverified.",
    "launcher_authority_unwired": "Real process manager does not transfer private authority handles/install a trusted factory.",
    "frontend_lifespan_unwired": "Actual API startup/native frontend binding/shutdown_close lifespan are not integrated.",
    "loaded_inventory_unverified": "No complete real loaded parser/renderer/tokenizer/model/executor/worker inventory proof.",
    "native_runtime_unverified": "No physical capture/publication, ASGI disconnect, completion or worker drain verification.",
    "peer_death_unwired": "Contract barrier never grants serving; native peer-death revocation remains unimplemented.",
}


class CandidateNotInstallable(RuntimeError):
    pass


def require_installable():
    # Environment alone grants nothing. Only the actual manager/engine private
    # bootstrap installs this process-local owner; route admission still needs F1.
    from remote_ra_linux.native_authority import current
    return current().verify_identity()


def candidate_enabled(environ=None):
    value = (os.environ if environ is None else environ).get("REMOTE_RA_DYNAMIC_CANDIDATE", "0")
    if value == "0":
        return False
    if value != "1":
        raise ValueError("REMOTE_RA_DYNAMIC_CANDIDATE must be 0 or 1")
    require_installable()
