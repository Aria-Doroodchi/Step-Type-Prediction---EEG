#!/usr/bin/env bash
# Sprint 2026-09-28, Phase 2: dress rehearsal of the release pipeline at the
# sealed data's realistic size (brief prompts/2026-09-28_release_readiness.md).
# Two resumable lanes on the full-size 120 Hz mock (14,400 windows, 43 EEG +
# 4 non-EEG channels, 20 subjects x 6 sessions x 2 contexts):
#   lane A (10 threads): the mock_sealed_s wcv_loso ablation step that did not
#     fit the build agents' 600 s cap, then train_sealed.sh end to end on
#     mock_sealed_120 with the harness weight, then with BLEND_W=auto
#   lane B (2 x 5 threads inside release_ablations.sh): a subset of the
#     release ablations on mock_sealed_120 under the RELEASE-DAY split
#     (SPLIT=calib:3, TEST_SUBJECTS = the 10 fully labelled participants, the
#     mock's hidden rows excluded; review finding R1-3: on replica:3 the
#     training set holds the full participants' decoupled sessions 3..5 and
#     masks the EMG trap). Per-step timing at full size + machinery rules.
# Every step is wrapped in /usr/bin/time -v (peak RSS in the step log).
#   wsl.exe bash -lc 'bash ~/codabench/scripts/sprint0928_p2.sh'   (LANES=A|B)
TAG=sprint0928_p2
export XS_THREADS=10
source "$HOME/codabench/scripts/sealed_lib.sh"
DH="$HOME/neuralbench/benchopt_data"
S="$HOME/codabench/scripts"
TIME=(/usr/bin/time -v)

laneA() {
  step s_wcv_loso 8 30m "${TIME[@]}" env TAG=sprint0928_abl_s STUDY=mock_sealed_s LANES=B \
      STEPS=wcv_loso XS_THREADS=10 bash "$S/release_ablations.sh"
  step t120 35 150m "${TIME[@]}" bash "$S/train_sealed.sh" "$DH" mock_sealed_120
  step t120_auto 70 210m "${TIME[@]}" env BLEND_W=auto bash "$S/train_sealed.sh" "$DH" mock_sealed_120
}
laneB() {
  step abl120 90 240m "${TIME[@]}" env TAG=sprint0928_abl120 STUDY=mock_sealed_120 XS_THREADS=5 \
      SPLIT=calib:3 TEST_SUBJECTS=0,1,2,3,4,5,6,7,8,9 \
      STEPS="base ch_all ch_eeg_emg ch_eeg_eog al_ctx xd0" bash "$S/release_ablations.sh"
}
LANES=${LANES:-AB}
[[ $LANES == *A* ]] && { laneA & }
[[ $LANES == *B* ]] && { laneB & }
wait
for s in s_wcv_loso t120 t120_auto abl120; do
  [ -f "$LOGDIR/$s.done" ] || { echo "| $(date +%T) | lanes $LANES finished; $s not done |" >> "$STATUS"; exit 0; }
done
echo "| $(date +%T) | ALL DONE |" >> "$STATUS"
