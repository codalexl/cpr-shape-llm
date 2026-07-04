# LLM Opponent Shaping Experiments
## Setup & Reproduction Playbook for M4 MacBook Pro (128GB)

**Goal**: Get you up and running quickly with the official `shape-llm` repository so you can reproduce the core results (especially Iterated Prisoner's Dilemma shaping) and then confidently branch into your own small experiments.

This playbook is tailored for **Apple Silicon M4 MacBook Pro with 128GB unified memory**. No CUDA. We use the **MPS (Metal Performance Shaders)** backend.

Focus for now: **Reproduction first → Understand the loop → Easy variant configs**. We keep experiments small, efficient, and many-runnable on your hardware.

---

## 1. Overview & Why This Starting Point

The paper introduces **ShapeLLM** — a model-free opponent shaping method for LLM agents using structured prompts + PPO fine-tuning.

Key setup in the repo:
- Small instruction-tuned model: `gemma-2-2b-it`
- LoRA (rank 2) + custom PPO
- Iterated matrix games (IPD, IMP, Chicken, etc.)
- Two main training modes:
  - `finetuning_fixed_opponent.py` (baseline)
  - `finetuning_two_learners.py` (used for shaping experiments — one agent acts as shaper with different update rhythm)

**Why start here?**
- Official code from the authors (https://github.com/martaemili/shape-llm)
- Clean, config-driven
- Small model = fast iteration on M4
- Once you can reproduce IPD shaping, you understand the core loop and can safely create variants

**Target for first successful run**: Reproduce shaping results on IPD (shaper reward ~3.9+, opponent ~0.1 or lower).

---

## 2. Hardware & Environment Considerations (M4 128GB)

**Good news**:
- Gemma-2-2b-it in bf16 is only ~2–3 GB
- With LoRA rank 2 + PPO, peak memory during training is very manageable on 128 GB unified memory
- M4 GPU cores + Neural Engine handle small-model training well

**Challenges & Solutions**:
- No CUDA → Use **MPS** backend
- `bitsandbytes` (used for QLoRA in paper) has **poor/no official MPS support** → We will run in **bf16 / fp16 without 4-bit quantization** (perfectly fine for 2B model on your machine)
- Custom PPO implementation (not the default TRL one at the time) → Works with MPS if we set device correctly
- Many small experiments → We will optimize for speed, low memory, and easy parallelization via configs + seeds

**Recommended Python env**: Use **Conda** (or `uv` / `venv` if you prefer). Clean isolation is important because we will modify torch install.

---

## 3. Step-by-Step Environment Setup (M4 Mac)

### 3.1 Create Clean Conda Environment

```bash
conda create -n shape-llm python=3.11 -y
conda activate shape-llm
```

### 3.2 Install PyTorch with MPS Support (Critical)

Do **NOT** use the cu121 version from the repo's requirements.

```bash
# Latest stable PyTorch with MPS support (as of mid-2026)
pip install torch torchvision torchaudio
```

Verify MPS works:

```bash
python -c "import torch; print(torch.__version__); print('MPS available:', torch.backends.mps.is_available()); print('MPS built:', torch.backends.mps.is_built())"
```

You should see `True` for both.

### 3.3 Install Remaining Dependencies

```bash
pip install transformers==4.47.0 \
            trl==0.11.4 \
            peft==0.14.0 \
            accelerate \
            datasets \
            wandb \          # highly recommended for logging
            matplotlib \
            pandas \
            pyyaml \
            psutil           # for memory monitoring
```

> **Note on bitsandbytes**: We skip the 4-bit version for now. If you later want to experiment with quantization on Apple Silicon, look into community solutions like `bitsandbytes-metal` or `torchao`, but it is unnecessary for the 2B model.

### 3.4 (Optional but Recommended) Install Git LFS if Needed

```bash
brew install git-lfs
git lfs install
```

---

## 4. Clone the Repository & Initial Exploration

```bash
git clone https://github.com/martaemili/shape-llm.git
cd shape-llm
```

### Key Files to Explore Immediately

```bash
ls -la
```

Important directories/files:
- `example_configs/` → `train_two_learners_config.json` (this is the shaping one), `train_fixed_opponent_config.json`, `eval_config.json`
- `finetuning_two_learners.py` → Main script for opponent shaping experiments
- `finetuning_fixed_opponent.py` → Baselines
- `agents.py`, `environment.py`, `observation_managers.py` → Core logic
- `evaluation_script.py`

**First action**: Read the two main training scripts and the example two-learners config (use `cat` or VS Code / Cursor).

---

## 5. Adapting the Code for Apple Silicon (MPS)

The original code was written for CUDA. You will likely need small edits.

### 5.1 Quick Device Detection Helper (Recommended)

Create a small utility or add at the top of scripts:

```python
import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")
```

### 5.2 Changes You Will Probably Need

1. In `finetuning_two_learners.py` and `agents.py`:
   - Replace hard-coded `device = "cuda"` with the `get_device()` logic above.
   - Make sure model loading uses `device_map` or `.to(device)` correctly.
   - For `torch_dtype`: Use `torch.bfloat16` (best on M4) or `torch.float16`.

2. In config files (JSON):
   - Add or modify a `"device": "mps"` field if the code reads it.
   - Set `"load_in_4bit": false` or remove 4-bit related keys (since we skip bitsandbytes 4-bit).

3. PPO / TRL parts:
   - The custom PPO should work. If you hit MPS-specific bugs (rare but possible with some ops), fall back to `device="cpu"` for the first few debugging runs (still fast for 2B model).

**Test a minimal load first** (before full training):

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = 'google/gemma-2-2b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto' if torch.backends.mps.is_available() else None
)
print('Model loaded successfully on', model.device)
"
```

---

## 6. Running the First Reproduction (IPD Shaping)

### 6.1 Choose the Right Config

Use `example_configs/train_two_learners_config.json` as starting point.

Copy it to create your first reproduction config:

```bash
cp example_configs/train_two_learners_config.json configs/ipd_shaping_repro.json
```

Edit `configs/ipd_shaping_repro.json`:
- Set game to IPD (if not already)
- Reduce number of trials / episodes for first test if you want faster feedback (e.g., 50–100 trials instead of 200–300)
- Make sure `"device": "mps"` or handle it in code
- Disable 4-bit quantization
- Set a clear `output_dir` and `wandb` project name

### 6.2 Launch Command

```bash
python finetuning_two_learners.py \
    configs/ipd_shaping_repro.json \
    checkpoints/ipd_shaping_repro \
    --n_seeds 1 \
    --no_epochs 1 \          # or adjust
    --checkpoint_freq 50
```

> The exact flags are shown in the repo README. Adjust `--n_seeds` and training length for your first runs.

### 6.3 Monitor While Running

- Use **Weights & Biases** (strongly recommended): `wandb login` then enable in config.
- Watch memory: Activity Monitor or `htop` (install via brew).
- In Python you can add:
  ```python
  import torch, psutil, os
  print(f"RAM used: {psutil.Process(os.getpid()).memory_info().rss / 1e9:.2f} GB")
  if torch.backends.mps.is_available():
      print(f"MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
  ```

### 6.4 Validation — Did It Work?

After training, run evaluation:

```bash
python evaluation_script.py configs/ipd_shaping_repro.json checkpoints/ipd_shaping_repro --seed 0
```

**Paper reference numbers for IPD shaping (approximate)**:
- Shaper average reward per step: ~3.96 ± 0.01
- Opponent: ~0.10 ± 0.04

If you get close to these (within reasonable variance for 1 seed + possibly shorter training), you have successfully reproduced the core result.

Also inspect learning curves (reward over trials) and state visitation plots if the code generates them.

---

## 7. Understanding the Training Loop (Key for Variants)

Spend time reading the code. Core concepts:

1. **Trial** = higher-level unit for the shaper (multiple parallel environments × several episodes).
2. **Episode** = one full game of T rounds (e.g., 20 rounds in IPD).
3. **Shaper vs Opponent asymmetry**:
   - Opponent (naive learner) updates its policy **after every episode**.
   - Shaper updates **only at the end of a full trial**, using cumulative reward across the whole trial. This lets it observe how the opponent is changing.
4. **Context / History** in prompts: Compressed visitation counts (`CC: 12, CD: 3, ...`) so the shaper can "see" the opponent's learning trajectory across episodes.
5. **Action mapping**: Specific tokens (e.g., `w_C`, `w_D`) or natural language mapped to game actions. Invalid tokens are penalized.

**Once you understand this asymmetry and the prompt context mechanism, creating variants becomes much easier.**

---

## 8. Creating & Running Variant Configs (Your Branching Point)

After successful reproduction:

### Simple Variant Workflow

1. Copy the working config:
   ```bash
   cp configs/ipd_shaping_repro.json configs/ipd_variant_myidea.json
   ```

2. Edit only the things you want to change:
   - Different game (change payoff matrix or game name)
   - Different model (try a slightly larger one if memory allows)
   - Different history representation (full trajectory vs counts vs LLM summary)
   - Change entropy regularization schedule
   - Different number of parallel envs (`n_games`)
   - Shorter/longer episodes
   - Pure in-context (no LoRA updates for shaper) — experimental

3. Run with a new output directory and a descriptive `wandb` run name.

4. Compare results side-by-side (use W&B groups or simple CSV logging).

**Recommended first variants (small changes)**:
- Different random seed (already easy with `--n_seeds`)
- Remove inter-episode history (test importance of context)
- Try a different matrix game from the paper (e.g., Chicken or Stag Hunt)
- Slightly higher/lower LoRA rank or learning rate
- Change temperature or sampling strategy

---

## 9. Efficiency Tips for Running *Lots* of Small Experiments on M4

- **Start small**: 50–100 trials, 1–2 seeds for quick iteration. Scale up only when promising.
- **Parallel experiments**: Use `tmux` or multiple terminal tabs. Or simple bash loop:
  ```bash
  for seed in 0 1 2; do
      python finetuning_two_learners.py configs/my_variant.json checkpoints/variant_s${seed} --n_seeds 1 --seed $seed &
  done
  ```
- **Memory discipline**: Close other apps. Monitor with `psutil`. If you ever hit limits, reduce batch size or context length first.
- **bf16 is your friend**: On M4, `torch.bfloat16` is fast and stable.
- **Checkpoint often** but not too often (saves time).
- **W&B** or local CSV logging so you can compare dozens of runs easily.
- **Profile one run** with `torch.profiler` or just time it, then optimize.

With 128 GB you can comfortably run 2–4 small experiments in parallel if needed.

---

## 10. Logging, Reproducibility & Good Habits

- Always use **W&B** (or equivalent) with clear tags: `game=IPD`, `variant=history_ablation`, `model=gemma-2-2b-it`, `seed=0`
- Log full config + git commit hash
- Save the exact prompt template used in every run
- For final important runs: fix all seeds (Python, numpy, torch, MPS)
- Keep a simple experiment tracking sheet (Google Sheet or Notion) with columns: config name, key hyperparams changed, shaper reward, opponent reward, notes, date

---

## 11. Validation Checklist (After First Few Runs)

- [ ] Model loads on MPS without error
- [ ] Training starts and makes progress (rewards change)
- [ ] No massive memory spikes or OOM
- [ ] Evaluation script runs and produces numbers
- [ ] Results directionally match paper (shaper >> opponent in IPD)
- [ ] You can explain why the shaper is updating on a different timescale than the opponent

---

## 12. Next Steps (Once Reproduction is Solid)

1. **Deepen understanding**: Read the paper sections on ShapeLLM and the ablations (especially history importance).
2. **Systematic small ablations**: History type, update frequency, prompt phrasing.
3. **New environments**: Add a simple new matrix game or text-based negotiation scaffold.
4. **Pure prompting baseline**: Turn off LoRA updates for the shaper and see how far in-context learning alone gets you.
5. **Scale slightly**: Try gemma-2-9b or Llama-3.1-8B if you want (still comfortable on 128 GB for short runs).

---

## 13. Troubleshooting (Common on Apple Silicon)

| Issue                        | Likely Cause                     | Fix / Workaround                              |
|-----------------------------|----------------------------------|-----------------------------------------------|
| `bitsandbytes` error        | No MPS support                   | Remove 4-bit / QLoRA, use bf16                |
| MPS op not implemented      | Some PyTorch ops missing       | Update torch or fall back to CPU for that op  |
| Slow training               | First run compilation          | Let it warm up; subsequent runs faster        |
| High memory usage           | Large context or batch         | Reduce `max_length`, batch size, or trials    |
| Reproducibility issues      | MPS non-determinism            | Use CPU for final critical runs or accept variance |
| Model download slow         | Hugging Face cache             | `export HF_HOME=~/huggingface_cache`          |

---

## 14. Resources

- Paper: https://arxiv.org/abs/2510.08255
- Official repo: https://github.com/martaemili/shape-llm
- This playbook lives at: `artifacts/LLM_Opponent_Shaping_M4_Setup_Playbook.md`

---

**You now have a clear, low-risk path to your first successful reproduction on your M4 MacBook Pro.**

Run the reproduction, get it working, understand the loop by reading the code while it trains, then start making small variant configs.

Once you have 1–2 solid reproduced runs + a couple of variants under your belt, come back and we can explore more ambitious directions (new domains, model-based shaping, negotiation environments, scaling studies, etc.) with much better intuition.

Ready when you are — paste any errors or questions here and we'll debug together.

Happy shaping! 🚀

---

*Playbook created for your M4 128GB setup — focused on fast, many small experiments and deep understanding before branching.*