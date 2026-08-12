#!/usr/bin/env bash
# All CPU regression suites. No GPU, no model load, no network.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-"uv run python"}

fail=0
for t in test_projection test_selectivity test_toolcalls test_channels test_eval_agentic test_eval_content test_eval_conversational; do
  echo "######## $t ########"
  $PY "scripts/$t.py" || fail=1
  echo
done

echo "######## corpus integrity ########"
$PY scripts/build_agentic_probes.py --check >/dev/null && echo "  agentic corpus matches source: OK" || { echo "  agentic corpus STALE — rebuild when no job is running"; fail=1; }
$PY scripts/build_conversational_probes.py --check >/dev/null && echo "  conversational corpus matches source: OK" || { echo "  conversational corpus STALE"; fail=1; }
$PY scripts/splits.py --audit-only >/dev/null && echo "  splits audit: OK" || fail=1

echo
[ $fail -eq 0 ] && echo "ALL SUITES PASSED" || echo "FAILURES PRESENT"
exit $fail
