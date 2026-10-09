# Two-city recovery: synthetic empty-fixture consistency

The original run stopped in `preflight-ist/no_projection/1`: its synthetic
fixture set one row's length and masks to zero but retained nonzero times.
`Batch._validate()` correctly rejected it with `AssertionError: wrong mask`.
This is a test-fixture error, not evidence of corrupt production data.

`tools/resume_two_city_emptyfix.py` adds an explicit recovery entry point; it
does not edit any file recorded in the original study's `code_sha256`.
The existing `manifest.json` remains authoritative for production parameters.

## Preservation and provenance

- All 30 NewYork outputs are rechecked against their hashes, source shards,
  indices, independently recomputed metrics, per-condition metrics, spatial RNG
  traces and saved W&B receipts. No NewYork generation/training is invoked.
- The original failure, registry, source manifest hash, recovery-code hashes,
  and hashes of preserved outputs/caches/preflights are recorded separately in
  `recovery/empty-fixture-v1/manifest.json`.
- The old invalid fixture, failed worker job/status/logs, successful normal
  preflight and all original source code remain in place.
- The corrected fixture is derived from the unchanged normal preflight cache.
  Only the synthetic row's length, masks, event times, intervals and per-event
  conditions are cleared. Context indicators and reference PO constraints are
  preserved, as are all other rows. The resulting Batch is validated.
- Synthetic GPU1 tests live under `recovery/empty-fixture-v1/preflight-ist`;
  normal GPU0 tests retain their old paths and reuse complete results.
- Production sampling/training code, batch sizes, seeds, optimizer budgets,
  temperature and CFG scale are unchanged. The full Istanbul preflight and
  GPU-equivalence check must pass before production sampling starts.
- Final completion requires both the original `audit.json` and the additional
  `recovery/empty-fixture-v1/audit.json` to pass. The second audit verifies that
  recovery code and all preserved artifacts still match their original hashes.

## Running and monitoring

The deployment launcher is
`calibration_runs/deployment/launch-two-city-emptyfix.sh`; the server copy is
`/root/experiments/pcdg/launch-two-city-emptyfix.sh`. It uses explicit `--resume`
and the original pipeline file lock to prevent concurrent controllers.

The active tmux session is `pcdg-two-city-v1`. Read
`/root/experiments/pcdg/two-city-v1-emptyfix-controller.log` for this recovery.
The older `two-city-v1-controller.log` and the original failed GPU1 status
are intentionally preserved historical evidence, not current progress.

The remaining sequence is unchanged: Istanbul ablations and non-CFG methods,
separate two-epoch CFG training preflight, fresh 1000-epoch / 110000-update CFG
training, CFG sampling for three seeds, audits, unified tables and delivery.
Never treat 30/60 or a passed preflight as a complete study.

Validation at deployment: five new local regression tests passed; all 84 server
tests passed. GPU-preflight progress must be read from the server receipts.
