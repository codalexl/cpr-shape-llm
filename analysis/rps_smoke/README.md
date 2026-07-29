# RPS shaping smoke analyses

Saved copies of Cursor canvases + numeric summaries so results survive outside the IDE canvas cache.

| Run | Config | Checkpoints | Canvas |
|---|---|---|---|
| Baseline (frozen shaper) | `configs/rps_shaping.json` | `checkpoints/rps_shaping_smoke/` | `canvases/rps-smoke-results.canvas.tsx` |
| Active (relaxed shaper) | `configs/rps_shaping_active.json` | `checkpoints/rps_shaping_active_smoke/` | `canvases/rps-smoke-active-results.canvas.tsx` |
| Active mid (100 ep) | `configs/rps_shaping_active.json` | `checkpoints/rps_shaping_active_mid/` | `canvases/rps-active-mid-results.canvas.tsx` |

Live canvases (open in Cursor beside chat) live under the project `canvases/` directory; these files are durable repo copies.

## Agent-2 hparam delta (active vs baseline)

| param | baseline | active |
|---|---|---|
| `learning_rate` | `1.41e-07` | `5e-7` |
| `cliprange` | `0.0001` | `0.1` |
| `vf_coef` | `0.001` | `0.05` |
| `init_entropy_coef` | `0.0` | `0.0` |

## Quick headline numbers

See `rps_smoke_summaries.json` for full per-epoch series.
