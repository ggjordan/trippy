# EXP-0007 — TRIPS on Hunua clip4982 (second scene from the brief)

Question: does the EXP-0003 recipe transfer to a 4K video clip scene (88 frames, sparse 75k
COLMAP points, Gaussian ply `clip4982_12000.ply`)?
Config: `config.yaml` (trips mode, knn sizes, 300 epochs, train_factor 1.0). Queued via
`scripts/queue_training.sh` so the run self-reports (held-out PSNR/SSIM/LPIPS, extent gate,
dolly + honesty sheet delivered). Shade-audit columns do not apply (no measured shade frames).
Gate: none of its own; informs v0.2.0 (does the method work beyond Karekare).
Numbers: pending.
