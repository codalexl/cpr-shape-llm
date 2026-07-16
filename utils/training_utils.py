import numpy as np 
import random 
import torch
import warnings
import time

from collections import defaultdict
from dataclasses import dataclass, field
from accelerate import Accelerator
from trl import PPOTrainer
from trl.core import clip_by_value, entropy_from_logits, flatten_dict, masked_mean, masked_var, masked_whiten, PPODecorators, convert_to_scalar, stats_to_np, stack_dicts, WANDB_PADDING, logprobs_from_logits
from trl.trainer.utils import get_global_statistics
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from utils.device_utils import get_device


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
        Updates running moments from batch's moments computed across ranks
        """
        if self.accelerator.use_distributed:
            xs_mean, xs_var, xs_count = get_global_statistics(self.accelerator, xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean, xs_var

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += (delta * xs_count / tot_count)
        new_var = tot_sum / tot_count
        self.std = (new_var * tot_count / (tot_count - 1)).sqrt()
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

