# Engineering Agent Notes: Sports2D PR #41

## 2026-11-14: Phase 1 Repository & Architecture Audit
- **Attempted**: Inspected branch `fix/temporal-lower-limb-swaps` (commit `3e57587` merging main), upstream diffs, `bilateral_body_pairs`, `correct_limb_swaps` in [`Sports2D/Utilities/common.py`](file:///home/mrudula/Downloads/mocap/Sports2D/Sports2D/Utilities/common.py), and invocation in [`Sports2D/process.py`](file:///home/mrudula/Downloads/mocap/Sports2D/Sports2D/process.py).
- **Observed**: Greedy online frame-by-frame correction mutates `prev_keypoints[person_idx]` upon swap. On limb crossover (e.g. sagittal crossing during gait/acrobatics), greedy cost swap updates baseline, locking in persistent false swaps or creating high-frequency label chatter.
- **Root Cause**: Feed-forward online decision lacks future trajectory evidence and cannot distinguish true identity swaps from natural biomechanical limb crossings without temporal lookahead or sequence regularization.
- **Do Not Repeat**: Do not lower thresholds further in online mode to "catch" swaps; lowering thresholds increases false-positive crossover triggers without solving persistence.
- **Next Action**: Build controlled evaluation framework and investigate offline 2-state Viterbi trellis operating on bilateral groups before interpolation.

## 2026-11-14: Phase 2-5 Viterbi Implementation & Verification
- **Attempted**: Formulated and implemented offline 2-state Viterbi filter in `Sports2D/Utilities/common.py` (`correct_limb_swaps_viterbi`), integrated into `Sports2D/process.py` before interpolation, restored missing `do_filter`/`reject_outliers` configs, fixed BlazePose `Lknee` case matching bug, and added safe import fallback for `get_face_keypoint_ids` and `blur_faces_in_frame`.
- **Observed**: 10/10 pytest tests pass in 0.27s; synthetic benchmark battery achieves 1.00 Precision, 1.00 Recall, 0 False Positives, and 0 Persistent Errors across all scenarios (including single-pair hallucination and sagittal crossing); end-to-end processing of `demo.mp4` completed in 36.05s.
- **Root Cause**: Offline sequence optimization with group-atomic swapping and state transition cost ($\lambda_{enter}=0.04$) naturally eliminates both persistent lock-in and single-pair hallucination swings without manual pixel thresholds.
- **Do Not Repeat**: Do not mutate temporal baseline in online feed-forward loops without lookahead; do not call `compute_height` without catching missing keypoint errors for partial skeleton models.
- **Documentation**: Comprehensive report written to [`Sports2D/docs/swap_correction_investigation.md`](file:///home/mrudula/Downloads/mocap/Sports2D/docs/swap_correction_investigation.md).

## 2026-11-14: Phase 6 Integration Regression & Reporting Reconciled
- **Attempted**: Verified existence of David's external clips (`swap3.mp4`, `swap_10019.mp4`), replaced trivial toggle test with end-to-end DataFrame + Butterworth filter stability integration test, and audited reporting discrepancy on frame 296.
- **Observed**: External clips `swap3.mp4` and `swap_10019.mp4` are absent locally (marked BLOCKED); 16/16 unit tests pass including filter stability (>99.9% acceleration spike reduction); the reporting discrepancy was resolved: 1 detector hallucination + greedy swap at frame 296 spread across 129 frames downstream due to Butterworth low-pass filter impulse response.
- **Root Cause**: Downstream filtering masks discrete swap events as continuous multi-frame shifts unless evaluated at the raw pre-filter stage.
- **Next Action**: Await local ingestion of `swap3.mp4` and `swap_10019.mp4` into `Sports2D/media/external_test_clips/` to run comparative benchmark.
