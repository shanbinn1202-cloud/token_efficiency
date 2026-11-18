#!/usr/bin/env bash
# Pre-pull and verify the SWE-bench evaluation images of the gswe32 task list on tbench (one-time; needs the user's
# approval for downloads). With --check-only nothing is pulled: it reports which images are present. Never prunes or
# removes any image (tbench holds other projects' images too). Prints one final JSON line.
# Usage: pull_images.sh [--check-only] [instance_id ...]   (default: the deployed tasks/gswe32-tasks.json)
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
CHECK=0
if [[ ${1:-} == --check-only ]]; then CHECK=1; shift; fi
INSTANCES=("$@")
if (( ${#INSTANCES[@]} == 0 )); then
  mapfile -t INSTANCES < <(hpy -c 'import json,sys; [print(i) for i in json.load(open(sys.argv[1]))["instances"]]' "$GSWE32_ROOT/tasks/gswe32-tasks.json")
fi
free_before=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc 0-9)
present=0 pulled=0 failed=() missing=()
for i in "${INSTANCES[@]}"; do
  image=$(image_of "$i")
  if docker image inspect "$image" > /dev/null 2>&1; then present=$((present + 1)); continue; fi
  if (( CHECK )); then missing+=("$image"); continue; fi
  if docker pull "$image" > /dev/null 2>&1 && docker image inspect "$image" > /dev/null 2>&1; then
    pulled=$((pulled + 1))
  else
    failed+=("$image")
  fi
done
free_after=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc 0-9)
hpy - "$present" "$pulled" "${free_before:-0}" "${free_after:-0}" "$CHECK" "${#INSTANCES[@]}" "${#failed[@]}" "${failed[@]}" "${missing[@]}" <<'PY'
import json, sys
present, pulled, before, after, check, total, nfailed, *rest = sys.argv[1:]
failed, missing = rest[:int(nfailed)], rest[int(nfailed):]
status = 'PASS' if not failed and not missing and int(present) + int(pulled) == int(total) else 'FAIL'
print(json.dumps(dict(event='gswe32_images', status=status, instances=int(total), present=int(present), pulled=int(pulled),
                      failed=failed, missing=missing, check_only=check == '1', free_gb_before=int(before),
                      free_gb_after=int(after)), sort_keys=True))
PY
