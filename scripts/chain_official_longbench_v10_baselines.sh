#!/bin/bash
set -euo pipefail

root=${OUTPUT_ROOT:-$(pwd)/proofkv_runs/acl_main_longbench_structured_20260724}
baseline_root="$root/official_baselines_v10"
ruler_audit="$root/official_ruler_v10/formal_audit_v10.json"
formal_protocol="$baseline_root/formal_protocol_v10.json"
q_program=proofkv_official_longbench_v10_qwen25
m_program=proofkv_official_longbench_v10_mistral02
q_preflight_program=proofkv_official_longbench_v10_preflight_qwen25
m_preflight_program=proofkv_official_longbench_v10_preflight_mistral02
tail_2wiki_program=proofkv_longbench_tail_mistral02_2wikimqa
tail_musique_program=proofkv_longbench_tail_mistral02_musique
q_preflight_dir="$baseline_root/preflight_qwen25_current"
m_preflight_dir="$baseline_root/preflight_mistral02_current"
preflight_audit="$baseline_root/current_code_smoke_preflight_audit_v10.json"
expected=940
preflight_expected=20

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

while ! artifact_passes "$ruler_audit"; do
  printf '%s waiting for official-RULER formal audit before loading official baseline models\n' \
    "$(date -Is)"
  sleep 60
done

for program in "$tail_2wiki_program" "$tail_musique_program"; do
  state=$(program_state "$program")
  if [[ "$state" == "RUNNING" || "$state" == "STARTING" ]]; then
    supervisorctl stop "$program"
  fi
done

printf '%s RULER formal audit passed; starting current-code official-baseline preflight\n' \
  "$(date -Is)"
for program in "$q_preflight_program" "$m_preflight_program"; do
  state=$(program_state "$program")
  if [[ "$state" == "STOPPED" || "$state" == "EXITED" || "$state" == "FATAL" ]]; then
    supervisorctl start "$program"
  fi
done

while true; do
  q_state=$(program_state "$q_preflight_program")
  m_state=$(program_state "$m_preflight_program")
  q_rows=$(row_count "$q_preflight_dir/results.jsonl")
  m_rows=$(row_count "$m_preflight_dir/results.jsonl")
  q_metrics=false
  m_metrics=false
  [[ -f "$q_preflight_dir/pilot_metrics_n1.json" ]] && q_metrics=true
  [[ -f "$m_preflight_dir/pilot_metrics_n1.json" ]] && m_metrics=true
  printf '%s official-baseline preflight qwen=%s/%s(%s,metrics=%s) mistral=%s/%s(%s,metrics=%s)\n' \
    "$(date -Is)" "$q_rows" "$preflight_expected" "$q_state" "$q_metrics" \
    "$m_rows" "$preflight_expected" "$m_state" "$m_metrics"
  if [[ "$q_rows" -eq "$preflight_expected" \
    && "$m_rows" -eq "$preflight_expected" \
    && "$q_state" == "EXITED" && "$m_state" == "EXITED" \
    && "$q_metrics" == true && "$m_metrics" == true ]]; then
    break
  fi
  if [[ ( "$q_rows" -lt "$preflight_expected" || "$q_metrics" != true ) \
    && ( "$q_state" == "STOPPED" || "$q_state" == "EXITED" \
      || "$q_state" == "FATAL" ) ]]; then
    supervisorctl start "$q_preflight_program"
  fi
  if [[ ( "$m_rows" -lt "$preflight_expected" || "$m_metrics" != true ) \
    && ( "$m_state" == "STOPPED" || "$m_state" == "EXITED" \
      || "$m_state" == "FATAL" ) ]]; then
    supervisorctl start "$m_preflight_program"
  fi
  sleep 15
done

/venv/main/bin/python scripts/audit_official_longbench_v10_smoke_preflight.py \
  --selection-manifest "$root/selection_manifest_v10_complement_n47.json" \
  --runner scripts/run_official_longbench_v10_baseline.py \
  --qwen-run-dir "$q_preflight_dir" \
  --mistral-run-dir "$m_preflight_dir" \
  --out "$preflight_audit"

artifact_passes "$preflight_audit"
printf '%s current-code official-baseline preflight audit passed\n' \
  "$(date -Is)"

/venv/main/bin/python scripts/freeze_official_longbench_v10_protocol.py \
  --selection-manifest "$root/selection_manifest_v10_complement_n47.json" \
  --controller "$root/domain_utility_controller_v10_control_only_complement.json" \
  --ruler-audit "$ruler_audit" \
  --runner scripts/run_official_longbench_v10_baseline.py \
  --audit scripts/audit_official_longbench_v10_baselines.py \
  --chain scripts/chain_official_longbench_v10_baselines.sh \
  --preflight-audit "$preflight_audit" \
  --preflight-audit-implementation \
    scripts/audit_official_longbench_v10_smoke_preflight.py \
  --formal-output-dir "$baseline_root/formal_qwen25" \
  --formal-output-dir "$baseline_root/formal_mistral02" \
  --bootstrap-replicates 50000 \
  --permutation-replicates 50000 \
  --seed 20260725 \
  --out "$formal_protocol"

printf '%s RULER formal audit passed; starting model-compatible official baselines\n' \
  "$(date -Is)"

for program in "$q_program" "$m_program"; do
  state=$(program_state "$program")
  if [[ "$state" == "STOPPED" || "$state" == "EXITED" || "$state" == "FATAL" ]]; then
    supervisorctl start "$program"
  fi
done

while true; do
  q_state=$(program_state "$q_program")
  m_state=$(program_state "$m_program")
  q_rows=$(row_count "$baseline_root/formal_qwen25/results.jsonl")
  m_rows=$(row_count "$baseline_root/formal_mistral02/results.jsonl")
  q_metrics=false
  m_metrics=false
  [[ -f "$baseline_root/formal_qwen25/metrics.json" ]] && q_metrics=true
  [[ -f "$baseline_root/formal_mistral02/metrics.json" ]] && m_metrics=true
  printf '%s official-baseline qwen=%s/%s(%s,metrics=%s) mistral=%s/%s(%s,metrics=%s)\n' \
    "$(date -Is)" "$q_rows" "$expected" "$q_state" "$q_metrics" \
    "$m_rows" "$expected" "$m_state" "$m_metrics"
  if [[ "$q_rows" -eq "$expected" && "$m_rows" -eq "$expected" \
    && "$q_state" == "EXITED" && "$m_state" == "EXITED" \
    && "$q_metrics" == true && "$m_metrics" == true ]]; then
    break
  fi
  if [[ "$q_rows" -lt "$expected" \
    && ( "$q_state" == "STOPPED" || "$q_state" == "EXITED" \
      || "$q_state" == "FATAL" ) ]]; then
    supervisorctl start "$q_program"
  fi
  if [[ "$m_rows" -lt "$expected" \
    && ( "$m_state" == "STOPPED" || "$m_state" == "EXITED" \
      || "$m_state" == "FATAL" ) ]]; then
    supervisorctl start "$m_program"
  fi
  sleep 30
done

printf '%s completed model-compatible official LongBench-v10 baselines\n' \
  "$(date -Is)"

longbench_audit="$root/audits/formal_full_v10_both_complete.json"
while ! artifact_passes "$longbench_audit"; do
  printf '%s waiting for the complete LongBench-v10 matrix audit\n' \
    "$(date -Is)"
  sleep 60
done

/venv/main/bin/python scripts/audit_official_longbench_v10_baselines.py \
  --selection-manifest "$root/selection_manifest_v10_complement_n47.json" \
  --formal-protocol "$formal_protocol" \
  --qwen-run-dir "$baseline_root/formal_qwen25" \
  --mistral-run-dir "$baseline_root/formal_mistral02" \
  --qwen-v10-run-dir "$root/formal_full_v10_qwen25_complement_n47" \
  --mistral-v10-run-dir "$root/formal_full_v10_mistral02_complement_n47" \
  --bootstrap-replicates 50000 \
  --permutation-replicates 50000 \
  --seed 20260725 \
  --out "$baseline_root/formal_audit_statistics.json"

artifact_passes "$baseline_root/formal_audit_statistics.json"
printf '%s official-baseline integrity audit and 50k paired inference passed\n' \
  "$(date -Is)"
