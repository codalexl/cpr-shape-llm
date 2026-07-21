import math
import numpy as np 
import random 
import torch
import warnings
import time

from collections import defaultdict
from dataclasses import dataclass, field
from accelerate import Accelerator
from transformers import LogitsProcessor
from trl import PPOTrainer
from trl.core import clip_by_value, entropy_from_logits, flatten_dict, masked_mean, masked_var, masked_whiten, PPODecorators, convert_to_scalar, stats_to_np, stack_dicts, WANDB_PADDING, logprobs_from_logits
from trl.trainer.utils import get_global_statistics
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from utils.device_utils import get_device


def mask_illegal_logits(logits: torch.Tensor, legal_tokens: List[int]) -> torch.Tensor:
    """Hard-mask logits so the policy support is exactly the legal action set."""
    banned = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
    banned[legal_tokens] = False
    # Prefer a large finite floor over -inf: safer for bf16 softmax / MPS sampling.
    return logits.masked_fill(banned, torch.finfo(logits.dtype).min)


def legal_action_logprobs(
    shift_logits: torch.Tensor,
    labels: torch.Tensor,
    legal_tokens: List[int],
) -> torch.Tensor:
    """
    Next-token log-probs under the categorical restricted to `legal_tokens`.

    `shift_logits` is [B, S, V] (already time-shifted); `labels` is [B, S].
    Positions whose label is not legal get 0 (caller must mask those out of the loss).
    """
    legal = torch.as_tensor(legal_tokens, device=shift_logits.device, dtype=torch.long)
    legal_logits = shift_logits.index_select(-1, legal)
    # Upcast softmax for numerical stability on bf16 / MPS, then cast back.
    log_sm = torch.log_softmax(legal_logits.float(), dim=-1).to(dtype=shift_logits.dtype)
    matches = labels.unsqueeze(-1) == legal.view(1, 1, -1)
    is_legal = matches.any(dim=-1)
    legal_index = matches.float().argmax(dim=-1)
    logp = log_sm.gather(-1, legal_index.unsqueeze(-1)).squeeze(-1)
    return torch.where(is_legal, logp, torch.zeros_like(logp))


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """Force generation to sample only from the legal action tokens."""

    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = list(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return mask_illegal_logits(scores, self.allowed_token_ids)


def entropy_from_masked_logits(logits: torch.Tensor, legal_tokens: List[int]) -> torch.Tensor:
    """Calculate entropy from logits restricted to legal action tokens."""
    pd = torch.nn.functional.softmax(logits[:, :, legal_tokens], dim=-1)
    entropy = torch.logsumexp(logits[:, :, legal_tokens], axis=-1) - torch.sum(
        pd * logits[:, :, legal_tokens], axis=-1
    )
    return entropy


@dataclass
class GradientTracker:
    keys: List[str] = field(default_factory=lambda: ["lora_A", "lora_B", "v_head_weight", "v_head_bias"])
    temp_grad_norm: Dict[str, List[float]] = field(init=False)

    def __post_init__(self):
        self.temp_grad_norm = {key: [] for key in self.keys}

    @torch.no_grad()
    def append_to_temp_grads(self, new_grads: Dict[str, torch.Tensor]):
        for key in self.keys:
            self.temp_grad_norm[key].append(new_grads[key])

@dataclass
class RunningMoments:
    accelerator: Accelerator
    mean: float = 0
    std: float = 1
    var: float = 1
    count: float = 1e-24

    @torch.no_grad()
    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """
        Updates running moments from batch's moments computed across ranks.

        Batches with fewer than 2 scores are skipped for the variance update so
        score scaling never hits the 0/0 path (std undefined for n < 2).
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        # Keep a usable scale factor when the filtered legal batch collapses.
        if xs_count < 2:
            return xs_mean, torch.ones((), device=xs.device, dtype=xs.dtype)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count)
        new_var = tot_sum / tot_count
        # tot_count starts near 0; guard the Bessel correction until we have ≥2 scores.
        bessel = max(tot_count - 1.0, 1.0)
        self.std = (new_var * tot_count / bessel).sqrt()
        self.var = new_var.item()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()

class CustomPPOTrainer(PPOTrainer):
    """Added functionality - considers full episode in GAE, and allows for non-zero masked entropy term"""
    def __init__(self, *args, track_gradients:bool=False, init_entropy_coef:float=0., final_entropy_coef:float=0., entropy_coef_horizon:int=1000, legal_tokens:Optional[List[int]]=None, a1_tok:int=235288, a2_tok:int=235299, a3_tok:Optional[int]=None, **kwargs): 
        super().__init__(*args, **kwargs)
        if legal_tokens is not None:
            self.legal_tokens = list(legal_tokens)
        else:
            self.legal_tokens = [a1_tok, a2_tok] + ([a3_tok] if a3_tok is not None else [])
        self.a1_tok, self.a2_tok = self.legal_tokens[0], self.legal_tokens[1]
        self.a3_tok = self.legal_tokens[2] if len(self.legal_tokens) > 2 else None
        self.env_ids, self.n = None, None 
        self.track_gradients = track_gradients

        # Save all the entropy control related parameters
        self.entropy_coef = init_entropy_coef # Initialise entropy to initial value 
        self.entropy_change_per_step = (final_entropy_coef - init_entropy_coef) / entropy_coef_horizon 
        self.final_entropy_coef = final_entropy_coef
        self.running = RunningMoments(self.accelerator) # Make sure we use custom RunningMoments 
        if self.track_gradients: 
            self.gradient_tracker = GradientTracker()

    def update_entropy_coef(self) -> None:
        """Updates the entropy coefficient in place"""
        self.entropy_coef = max(self.final_entropy_coef, self.entropy_coef + self.entropy_change_per_step)


    def batched_forward_pass(
        self,
        model,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        """Same as TRL's forward pass, with legal-set log-probs only on response tokens.

        Important: do NOT hard-mask the full vocabulary on prompt positions. That makes
        prompt log-probs -inf; then (-inf)-(-inf) → NaN in the PPO ratio, and
        masked_mean cannot recover (NaN*0 is still NaN).
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            logits, _, values = model(**input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            shift_logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            # Unconstrained logprobs on the prompt; constrained on the action tokens.
            unconstrained = logprobs_from_logits(shift_logits, labels)
            constrained = legal_action_logprobs(shift_logits, labels, self.legal_tokens)

            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1
                    if attention_mask[j, 0] == 0:
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j]

            response_mask = masks[:, :-1].bool()
            logprobs = torch.where(response_mask, constrained, unconstrained)

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def compute_advantages(self, values: torch.FloatTensor, rewards: torch.FloatTensor, mask: torch.FloatTensor,):
        """Custom function to compute advantages - Multi-Turn PPO"""
        values = values * mask
        rewards = rewards * mask

        if self.config.whiten_rewards:
            rewards = masked_whiten(rewards, mask, shift_mean=False)
            rewards = torch.masked_fill(rewards, ~mask.bool(), 0) # Added this because of the latest version of TRL 

        advantages = torch.zeros(rewards.shape)
        time_steps = int(rewards.shape[0] / self.n)

        for game_id in range(self.n): 
          # Get positions within flattened rewards
          target_positions = torch.where(torch.tensor(self.env_ids) == game_id)[0]
          time_steps = len(target_positions)
          for t in reversed(range(time_steps)): 
            v_next = values[target_positions[t+1],-1] if t < (time_steps-1) else 0.0 # Extract next values
            delta = rewards[target_positions[t], -1] + self.config.gamma * v_next - values[target_positions[t], -1]
            adv_next = advantages[target_positions[t+1], -1] if t < (time_steps-1) else 0.0
            advantages[target_positions[t], -1] = delta + self.config.gamma * self.config.lam * adv_next
        advantages = advantages.to(get_device())

        returns = advantages + values
        advantages = masked_whiten(advantages, mask) # (* mask) This was the previous version 
        advantages = torch.masked_fill(advantages, ~mask.bool(), 0) # Otherwise advantages are not really zero. # Added this 
        advantages = advantages.detach()

        return values, advantages, returns
    
    @PPODecorators.empty_device_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
        advantages: torch.FloatTensor,
        returns: torch.FloatTensor,
    ):
        """Train PPO minibatch. Identical to TRL Trainer version except for the gradient tracking and the gradient clipping (the original line does not work)."""
        self.model.train()
        loss_p, loss_v, train_stats = self.loss(
            old_logprobs, values, logits, vpreds, logprobs, mask, advantages, returns
        )
        loss = loss_p + loss_v
        self.accelerator.backward(loss)
        if self.config.max_grad_norm is not None:
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

 
        if self.track_gradients:
            lora_A, lora_B, lora_A_count, lora_B_count = 0.0, 0.0, 0, 0
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if "lora_A" in name:
                        lora_A += param.grad.norm().item()
                        lora_A_count += 1
                    elif "lora_B" in name:
                        lora_B += param.grad.norm().item()
                        lora_B_count += 1
                    elif "v_head.summary.weight" in name:
                        v_head_weight = param.grad.norm().item()
                    elif "v_head.summary.bias" in name:
                        v_head_bias = param.grad.norm().item()
            grad_norms = {"lora_A": lora_A/lora_A_count, "lora_B": lora_B/lora_B_count, "v_head_weight": v_head_weight, "v_head_bias": v_head_bias}
            self.gradient_tracker.append_to_temp_grads(grad_norms)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return train_stats
    
    def loss(self, old_logprobs: torch.FloatTensor, values: torch.FloatTensor, logits: torch.FloatTensor, vpreds: torch.FloatTensor, logprobs: torch.FloatTensor, \
        mask: torch.LongTensor, advantages: torch.FloatTensor, returns: torch.FloatTensor):
        """Added masked entropy term to loss function"""
        vpredclipped = clip_by_value(vpreds, values - self.config.cliprange_value, values + self.config.cliprange_value,)

        vf_losses1 = (vpreds - returns) ** 2
        vf_losses2 = (vpredclipped - returns) ** 2
        vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)
        vf_clipfrac = masked_mean(torch.gt(vf_losses2, vf_losses1).float(), mask)

        ratio = torch.exp(logprobs - old_logprobs)

        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)

        pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)
        pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses).float(), mask)

        entropy = masked_mean(entropy_from_masked_logits(logits, self.legal_tokens), mask) # Before calculated masked mean of full logits
        loss = pg_loss + self.config.vf_coef * vf_loss - self.entropy_coef * entropy

        avg_ratio = masked_mean(ratio, mask).item()
        if avg_ratio > self.config.ratio_threshold:
            warnings.warn(
                f"The average ratio of batch ({avg_ratio:.2f}) exceeds threshold {self.config.ratio_threshold:.2f}. Skipping batch."
            )
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0
            loss = loss * 0.0

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(old_logprobs - logprobs, mask)

        return_mean, return_var = masked_mean(returns, mask), masked_var(returns, mask)
        value_mean, value_var = masked_mean(values, mask), masked_var(values, mask)

        stats = dict(
            loss=dict(policy=pg_loss.detach(), value=vf_loss.detach(), total=loss.detach()),
            policy=dict(
                entropy=entropy.detach(),
                approxkl=approxkl.detach(),
                policykl=policykl.detach(),
                clipfrac=pg_clipfrac.detach(),
                advantages=advantages.detach(),
                advantages_mean=masked_mean(advantages, mask).detach(),
                ratio=ratio.detach(),
            ),
            returns=dict(mean=return_mean.detach(), var=return_var.detach()),
            val=dict(
                vpred=masked_mean(vpreds, mask).detach(),
                error=masked_mean((vpreds - returns) ** 2, mask).detach(),
                clipfrac=vf_clipfrac.detach(),
                mean=value_mean.detach(),
                var=value_var.detach(),
            ),
        )
        return pg_loss, self.config.vf_coef * vf_loss - self.entropy_coef * entropy, flatten_dict(stats) # Before no entropy term returned here



def set_seed(seed:int) -> None:
    """Sets global seed to a given number"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_trainable_params(model, unfrozen_layer_IDs:List=[10, 11]):
    """Freezes all parameters of a given model except for the specified layers. Value and Language Modelling Heads are always unfrozen"""
    unfrozen_layers = ["v_head"] + ["h.{}".format(x) for x in unfrozen_layer_IDs]
    for name, param in model.named_parameters():
        param.requires_grad = True if (param.dtype.is_floating_point and (any(layer in name for layer in unfrozen_layers))) else False 
    model.pretrained_model.lm_head.weight.requires_grad = True  # Unfreeze LM head

