#!/bin/bash
set -euo pipefail

repo=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
root=${OUTPUT_ROOT:-$(cd "$repo/../.." && pwd)/proofkv_runs/acl_main_longbench_structured_20260724}
baseline_root="$root/official_baselines_v10"
ruler_audit="$root/official_ruler_v10/formal_audit_v10.json"
formal_protocol="$baseline_root/formal_protocol_v10_state_reset_r1.json"
preflight_audit="$baseline_root/current_code_state_reset_preflight_audit_r1.json"
q_preflight_dir="$baseline_root/preflight_qwen25_state_reset_r1"
m_preflight_dir="$baseline_root/preflight_mistral02_state_reset_r1"
q_formal_dir="$baseline_root/formal_qwen25_state_reset_r1"
m_formal_dir="$baseline_root/formal_mistral02_state_reset_r1"
q_preflight_program=proofkv_official_longbench_v10_preflight_qwen25_state_reset_r1
m_preflight_program=proofkv_official_longbench_v10_preflight_mistral02_state_reset_r1
q_program=proofkv_official_longbench_v10_qwen25_state_reset_r1
m_program=proofkv_official_longbench_v10_mistral02_state_reset_r1
tail_chain=proofkv_longbench_v10_tail_shard_chain
tail_2wiki=proofkv_longbench_tail_mistral02_2wikimqa
tail_musique=proofkv_longbench_tail_mistral02_musique
preflight_expected=20
formal_expected=940

row_count() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  /venv/main/bin/python - "$path" <<'PY'
import json
import sys

count = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        json.loads(line)
        count += 1
print(count)
PY
}

program_state() {
  local output
  output=$(supervisorctl status "$1" 2>/dev/null || true)
  awk '{print $2}' <<<"$output"
}

is_terminal_state() {
  local state=$1
  [[ "$state" == "EXITED" || "$state" == "STOPPED" ]]
}

artifact_passes() {
  local path=$1
  [[ -f "$path" ]] || return 1
  /venv/main/bin/python - "$path" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if payload.get("status") == "PASS" else 1)
PY
}

start_if_incomplete() {
  local program=$1
  local run_dir=$2
  local expected=$3
  local rows
  local state
  rows=$(row_count "$run_dir/results.jsonl")
  state=$(program_state "$program")
  if [[ "$rows" -lt "$expected" \
    && ( "$state" == "STOPPED" || "$state" == "EXITED" \
      || "$state" == "FATAL" ) ]]; then
    supervisorctl start "$program"
  fi
}

while ! artifact_passes "$ruler_audit"; do
  printf '%s state-reset official chain waits for RULER audit\n' "$(date -Is)"
  sleep 60
done

# Reserve exactly the two slots released by the invalidated official jobs.
for program in "$tail_chain" "$tail_2wiki" "$tail_musique"; do
  state=$(program_state "$program")
  if [[ "$state" == "RUNNING" || "$state" == "STARTING" ]]; then
    supervisorctl stop "$program"
  fi
done

printf '%s starting state-reset cross-capacity preflight\n' "$(date -Is)"
start_if_incomplete \
  "$q_preflight_program" "$q_preflight_dir" "$preflight_expected"
start_if_incomplete \
  "$m_preflight_program" "$m_preflight_dir" "$preflight_expected"

while true; do
  q_rows=$(row_count "$q_preflight_dir/results.jsonl")
  m_rows=$(row_count "$m_preflight_dir/results.jsonl")
  q_state=$(program_state "$q_preflight_program")
  m_state=$(program_state "$m_preflight_program")
  q_metrics=false
  m_metrics=false
  [[ -f "$q_preflight_dir/pilot_metrics_n1.json" ]] && q_metrics=true
  [[ -f "$m_preflight_dir/pilot_metrics_n1.json" ]] && m_metrics=true
  printf '%s state-reset preflight q=%s/%s(%s,%s) m=%s/%s(%s,%s)\n' \
    "$(date -Is)" \
    "$q_rows" "$preflight_expected" "$q_state" "$q_metrics" \
    "$m_rows" "$preflight_expected" "$m_state" "$m_metrics"
  if [[ "$q_rows" -eq "$preflight_expected" && "$q_metrics" == true \
    && "$m_rows" -eq "$preflight_expected" && "$m_metrics" == true ]] \
    && is_terminal_state "$q_state" \
    && is_terminal_state "$m_state"; then
    break
  fi
  start_if_incomplete \
    "$q_preflight_program" "$q_preflight_dir" "$preflight_expected"
  start_if_incomplete \
    "$m_preflight_program" "$m_preflight_dir" "$preflight_expected"
  sleep 15
done

cd "$repo"
/venv/main/bin/python scripts/audit_official_longbench_v10_smoke_preflight.py \
  --selection-manifest "$root/selection_manifest_v10_complement_n47.json" \
  --runner scripts/run_official_longbench_v10_baseline.py \
  --qwen-run-dir "$q_preflight_dir" \
  --mistral-run-dir "$m_preflight_dir" \
  --out "$preflight_audit"
artifact_passes "$preflight_audit"

/venv/main/bin/python scripts/freeze_official_longbench_v10_protocol.py \
  --selection-manifest "$root/selection_manifest_v10_complement_n47.json" \
  --controller \
    "$root/domain_utility_controller_v10_control_only_complement.json" \
  --ruler-audit "$ruler_audit" \
  --runner scripts/run_official_longbench_v10_baseline.py \
  --audit scripts/audit_official_longbench_v10_baselines.py \
  --chain scripts/chain_official_longbench_v10_baselines_state_reset_r1.sh \
  --preflight-audit "$preflight_audit" \
  --preflight-audit-implementation \
    scripts/audit_official_longbench_v10_smoke_preflight.py \
  --formal-output-dir "$q_formal_dir" \
  --formal-output-dir "$m_formal_dir" \
  --bootstrap-replicates 50000 \
  --permutation-replicates 50000 \
  --seed 20260725 \
  --out "$formal_protocol"

printf '%s state-reset protocol frozen; starting formal official jobs\n' \
  "$(date -Is)"
start_if_incomplete "$q_program" "$q_formal_dir" "$formal_expected"
start_if_incomplete "$m_program" "$m_formal_dir" "$formal_expected"

while true; do
  q_rows=$(row_count "$q_formal_dir/results.jsonl")
  m_rows=$(row_count "$m_formal_dir/results.jsonl")
  q_state=$(program_state "$q_program")
  m_state=$(program_state "$m_program")
  q_metrics=false
  m_metrics=false
  [[ -f "$q_formal_dir/metrics.json" ]] && q_metrics=true
  [[ -f "$m_formal_dir/metrics.json" ]] && m_metrics=true
  printf '%s state-reset formal q=%s/%s(%s,%s) m=%s/%s(%s,%s)\n' \
    "$(date -Is)" \
    "$q_rows" "$formal_expected" "$q_state" "$q_metrics" \
    "$m_rows" "$formal_expected" "$m_state" "$m_metrics"
  if [[ "$q_rows" -eq "$formal_expected" && "$q_metrics" == true \
    && "$m_rows" -eq "$formal_expected" && "$m_metrics" == true ]] \
    && is_terminal_state "$q_state" \
    && is_terminal_state "$m_state"; then
    break
  fi
  start_if_incomplete "$q_program" "$q_formal_dir" "$formal_expected"
  start_if_incomplete "$m_program" "$m_formal_dir" "$formal_expected"
  sleep 30
done

if [[ "$(program_state "$tail_chain")" != "RUNNING" ]]; then
  supervisorctl start "$tail_chain"
fi
printf '%s state-reset official jobs complete; restored tail schedule\n' \
  "$(date -Is)"

longbench_audit="$root/audits/formal_full_v10_both_complete.json"
while ! artifact_passes "$longbench_audit"; do
  printf '%s state-reset official audit waits for LongBench-v10 audit\n' \
    "$(date -Is)"
  sleep 60
done

/venv/main/bin/python scripts/audit_official_longbench_v10_baselines.py \
  --selection-manifest "$root/selection_manifest_v10_complement_n47.json" \
  --formal-protocol "$formal_protocol" \
  --qwen-run-dir "$q_formal_dir" \
  --mistral-run-dir "$m_formal_dir" \
  --qwen-v10-run-dir "$root/formal_full_v10_qwen25_complement_n47" \
  --mistral-v10-run-dir "$root/formal_full_v10_mistral02_complement_n47" \
  --bootstrap-replicates 50000 \
  --permutation-replicates 50000 \
  --seed 20260725 \
  --out "$baseline_root/formal_audit_statistics_state_reset_r1.json"

artifact_passes \
  "$baseline_root/formal_audit_statistics_state_reset_r1.json"
printf '%s state-reset official integrity and paired inference passed\n' \
  "$(date -Is)"
