# shape-llm

This repository contains implementation and analysis code for the paper *Opponent Shaping in LLM Agents*, ICLR'26.

## Overview

This repository provides a framework for training LLM agents with PPO to play iterated matrix games (e.g. Prisoner's Dilemma). We provide various training setups: agents can be trained against fixed opponents (TFT, random), or two LLM learners can be trained simultaneously against each other.

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Install PyTorch separately with the CUDA version matching your hardware, e.g.:
> ```bash
> pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
> ```

## Usage

All scripts take a JSON config file as input. Example configs are provided in [`example_configs/`](example_configs/).

**Train against a fixed opponent:**
```bash
python finetuning_fixed_opponent.py <config_path> <saving_path> [--n_seeds N] [--no_epochs N] [--checkpoint_freq N]
```

**Train two LLM agents against each other:**
```bash
python finetuning_two_learners.py <config_path> <saving_path> [--n_seeds N] [--no_epochs N] [--checkpoint_freq N]
```

**Evaluate trained agents:**
```bash
python evaluation_script.py <config_path> <saving_path> [--seed N]
```
