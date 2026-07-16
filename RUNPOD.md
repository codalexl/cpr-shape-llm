# RunPod Instruction Book (personal notes — do not commit)

> This file is intentionally **untracked**. It will show up as `?? RUNPOD.md` in
> `git status`. Leave it that way, or add `RUNPOD.md` to `.gitignore`.

---

## TL;DR — Stop vs Terminate vs GPU upgrade

| Action | Keeps `/workspace`? | Can change GPU? | Billing |
|---|---|---|---|
| **Stop** | Yes (same pod disk) | **No** | Cheap storage while stopped |
| **Terminate** | **No** (gone unless Network Volume) | N/A (pod deleted) | Stops compute |
| **New pod** | Fresh disk (unless Network Volume) | Yes — pick any GPU | Full rate while Running |

- You **cannot change GPU type** on a stopped pod. Upgrade = **new pod**.
- Attach a **Network Volume** if you want repo + checkpoints to survive terminate/recreate.
- Before Terminate without a volume: `tar` / `rsync` checkpoints out.

---

## OOM on this codebase (important)

Two Gemma-2-2B + LoRA + PPO value heads in **bf16** sit on **one GPU**. Peak VRAM is during `loss.backward()` with `mini_batch_size`.

Your crash on a **~45 GB** card:

```text
Tried to allocate 3.50 GiB. ... 44.52 GiB total ... 892 MiB free
... 38.68 GiB allocated by PyTorch
```

A bigger GPU probably *would* fix this specific crash — you were only ~4 GiB short, which is
roughly the size of one mini-batch's logits tensor (256k Gemma vocab). But that also means
lowering `mini_batch_size` should close the same gap for free, on the card you already have.
**Try the free fix before paying for a bigger card.**

### Memory knobs (`configs/rps_shaping.json`)

| Knob | Default | Safer for ≤48 GB |
|---|---|---|
| `ppo_params.mini_batch_size` (both agents) | `10` | **`5` — already committed in the repo** |
| `ppo_params.batch_size` | `100` | leave for now (affects steps, not peak as hard) |
| `game_parameters.n_games` / `e_max` / `t_max` | 5 / 5 / 20 | only if still OOM |

`mini_batch_size: 5` is already checked into `configs/rps_shaping.json` — no sed needed on a
fresh pod, just re-clone/re-sync and it's there. Verify with:

```bash
grep mini_batch_size configs/rps_shaping.json
```

If you ever need to go lower manually (e.g. `5` -> `2`):

```bash
sed -i 's/"mini_batch_size": 5/"mini_batch_size": 2/g' configs/rps_shaping.json
```

Optional fragment helper:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Then re-run:

```bash
./scripts/run_rps_shaping.sh smoke
```

If still OOM: try `mini_batch_size: 1` + a **80 GB** card (A100 80GB / H100). Do **not** expect a 24 GB card to work with current defaults.

Theory note: `mini_batch_size` only changes how many/how noisy the SGD steps are within a PPO
epoch — GAE/advantages/rewards are computed once over the full `batch_size` regardless, so
lowering it doesn't change the RL objective, just optimization granularity. Worth a line in the
thesis methodology if used on the `_repro` config, since it deviates from the paper's exact
hyperparameters (gradient clipping and PPO-ratio staleness both get slightly noisier/more
frequent with smaller mini-batches).

---

## 0) Create pod (RunPod UI)

- **GPU:** prefer **48 GB+** (A6000 / L40 / A100 40–80). After lowering `mini_batch_size`, 40–48 GB is usually enough.
- **Container disk:** 50–100 GB
- **Network Volume:** strongly recommended (mount at e.g. `/workspace`)
- **Template:** PyTorch / CUDA
- Deploy → wait until **Running**

---

## 1) SSH from your Mac

```bash
ssh <user>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

(Use the exact Connect → SSH line from the RunPod UI. Space after `-i`.)

Checks:

```bash
nvidia-smi
df -h
```

---

## 2) Get the code onto the pod

Private repo — GitHub **rejects account passwords**. Use a **PAT** as the password, or `rsync` from your Mac.

### Option A — clone with PAT

```bash
cd /workspace   # or your Network Volume mount
git clone https://github.com/codalexl/cpr-shape-llm.git
cd cpr-shape-llm
# Username: codalexl
# Password: <GitHub personal access token with repo scope>
```

### Option B — rsync from Mac (no GitHub auth on pod)

```bash
# on Mac, from repo root
rsync -avz -e "ssh -i ~/.ssh/id_ed25519" \
  --exclude '.git' --exclude 'checkpoints' --exclude '__pycache__' \
  ./ <user>@ssh.runpod.io:/workspace/cpr-shape-llm/
```

---

## 3) Install deps

```bash
cd /workspace/cpr-shape-llm
python -m pip install -U pip
pip install -r requirements.txt

# optional: align torch/vision/audio (ignore torchvision/torchaudio warnings otherwise)
pip install --upgrade --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

`torchvision` / `torchaudio` version mismatch warnings after install are usually **harmless** for this training code.

---

## 4) Hugging Face (Gemma)

Accept the Gemma license on the Hub once, then:

```bash
huggingface-cli login
# or: export HF_TOKEN=...
```

Adapters auto-create on first run via `init_lora_adapters.py` if missing.

---

## 5) Apply OOM-safe config, then smoke

```bash
# mini_batch_size: 5 is already committed — just verify:
grep mini_batch_size configs/rps_shaping.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# watch VRAM in another SSH session: watch -n1 nvidia-smi
./scripts/run_rps_shaping.sh smoke
```

Success look: finishes epochs without `CUDA out of memory`; checkpoints under `checkpoints/rps_shaping_smoke/`.

---

## 6) Longer run

```bash
./scripts/run_rps_shaping.sh mid
# or
./scripts/run_ipd_shaping.sh full   # also lower mini_batch_size in that config
```

Use `tmux` (or `nohup`) so SSH disconnects don’t kill training.

**Some RunPod base images don’t ship tmux** — install it first:

```bash
apt-get update && apt-get install -y tmux
```

```bash
tmux new -s train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
./scripts/run_rps_shaping.sh mid 2>&1 | tee mid_run.log
# detach: Ctrl-b then d
# reattach: tmux attach -t train
```

**If `apt-get` isn’t available** (minimal image, no root package manager), fall back to
`nohup` — no detach/reattach, but it survives SSH disconnects the same way:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup ./scripts/run_rps_shaping.sh mid > mid_run.log 2>&1 &
disown
# check progress any time:
tail -f mid_run.log
```

---

## 7) Before Terminate (no Network Volume)

```bash
cd /workspace/cpr-shape-llm
tar -czf checkpoints.tgz checkpoints
```

From Mac:

```bash
scp -i ~/.ssh/id_ed25519 -P <PORT> \
  root@<IP>:/workspace/cpr-shape-llm/checkpoints.tgz .
```

Or rsync the `checkpoints/` folder.

---

## Stack health check

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); import bitsandbytes as bnb; print('bnb', bnb.__version__)"
```

---

## Checklist when moving to a new pod (your current situation)

1. Stop/Terminate the old pod (pull checkpoints first if you care about them).
2. Deploy new pod (48 GB+ preferred).
3. SSH in → `nvidia-smi`.
4. Clone / rsync repo.
5. `pip install -r requirements.txt`.
6. `huggingface-cli login` if needed.
7. Confirm `mini_batch_size: 5` in the config you will run (already committed for RPS).
8. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
9. `apt-get install -y tmux` (or skip and use `nohup` — see §6) then run detached.
10. `./scripts/run_rps_shaping.sh smoke`
11. Only then start `mid` / `full`.
