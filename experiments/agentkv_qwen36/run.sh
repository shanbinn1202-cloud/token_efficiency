#!/usr/bin/env bash
exec bash "$(dirname "${BASH_SOURCE[0]}")/../../scripts/run.sh" --method agentkv "$@"
