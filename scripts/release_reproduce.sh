#!/usr/bin/env bash
# Run every benchmark FROM THE RELEASE TREE, seeds 0 and 1 (the code LLM's seed):
#   mindcube: 3 slices     mmsi: all 1000     force3d: REF + SAG, non-GT
# Every question is planned and its program written through the code LLM; each run's provider log
# records the served model of every call. Score with scripts/release_score.py.
#   REL=<release root> VLM_PORT=<vllm port> bash scripts/release_reproduce.sh [mindcube|mmsi|force3d ...]
# A run is one benchmark and seed (one saturn.cli process). Exit status 0 means every run's process
# finished; it does not mean every question was answered or correct (that is in the results files).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)   # the release itself, or the source tree that builds it into release/saturn
REL=${REL:-$([ -d "$ROOT/release/saturn" ] && echo "$ROOT/release/saturn" || echo "$ROOT")}; cd "$REL" || exit 1
export SAPY_VLM_PORT=${VLM_PORT:-8100} SAPY_SERVE_HTTP_PORT=${SAPY_SERVE_HTTP_PORT:-8011}
mkdir -p experiments
RUNS=0; ERRORED_RUNS=()   # runs started; runs whose process exited with a non-zero status
run() { local tag=$1; shift; export SAPY_PROVIDER_LOG=$REL/experiments/provider_log_${tag}.jsonl
        echo "################ [$tag] $* ################  $(date '+%F %T')"; "$@"; local rc=$?
        RUNS=$((RUNS + 1)); echo "[$tag] exited with status $rc at $(date '+%T')"
        [ "$rc" -eq 0 ] || ERRORED_RUNS+=("$tag (status $rc)"); }
want() { [ $# -eq 0 ] && return 0; for p in "$@"; do [ "$p" = "$SEL" ] && return 0; done; return 1; }
SEL_ALL="$*"
for SEL in mindcube mmsi force3d; do
  want $SEL_ALL || continue
  case $SEL in
    mindcube) for seed in 0 1; do for s in among around rotation; do run mc_${s}_s$seed scripts/run_mindcube.sh $s $seed; done; done ;;
    mmsi)     for seed in 0 1; do run mmsi_s$seed scripts/run_mmsi.sh $seed; done ;;
    force3d)  for seed in 0 1; do run ref_nogt_s$seed scripts/run_force3d.sh ref $seed; run sag_nogt_s$seed scripts/run_force3d.sh puzzle $seed; done ;;
  esac
done
echo "code-LLM calls per run:"; for f in experiments/provider_log_*.jsonl; do [ -f "$f" ] && echo "  $(basename $f): $(wc -l < $f)"; done
if [ ${#ERRORED_RUNS[@]} -gt 0 ]; then
  (IFS=,; echo "Runs that exited with an error: ${ERRORED_RUNS[*]}" | sed 's/,/, /g') >&2; exit 1
fi
echo "All $RUNS runs exited with status 0. Per-question results are in experiments/; score them with scripts/release_score.py."
