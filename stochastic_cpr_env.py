"""
Stochastic Common-Pool Resource (CPR) Environment
Clean, self-contained implementation for LLM agent experiments
(inspired by classic CPR models + 2017 NeurIPS ideas, built from scratch)

Key features:
- Shared renewable resource with logistic growth + stochastic noise
- Discrete harvest actions (easy for LLM token output)
- Supports 2+ agents
- Partial observability ready
- Easy to add communication, shocks, or institutional actions later

Usage:
    env = StochasticCPR(n_agents=2, max_steps=100)
    obs = env.reset()
    actions = [2, 1]   # harvest amounts
    next_obs, rewards, done, info = env.step(actions)
"""

import numpy as np
from typing import List, Dict, Any, Tuple


class StochasticCPR:
    def __init__(
        self,
        n_agents: int = 2,
        S_max: float = 100.0,
        initial_S_range: Tuple[float, float] = (65.0, 90.0),
        r: float = 0.18,           # intrinsic growth rate
        K: float = 100.0,          # carrying capacity
        noise_std: float = 5.0,    # stochastic shock strength
        max_harvest: int = 8,      # max discrete harvest per agent per step
        max_steps: int = 120,
        seed: int = None
    ):
        """
        Initialize the stochastic common-pool resource environment.
        
        This creates a shared renewable resource that agents harvest from.
        The resource grows logistically but is subject to stochastic noise.
        Over-harvesting can deplete it (tragedy of the commons risk).
        """
        self.n_agents = n_agents
        self.S_max = S_max
        self.initial_S_range = initial_S_range
        self.r = r
        self.K = K
        self.noise_std = noise_std
        self.max_harvest = max_harvest
        self.max_steps = max_steps
        
        self.rng = np.random.default_rng(seed)
        
        # State variables
        self.S: float = 0.0
        self.step_count: int = 0
        self.history: List[Dict] = []   # for debugging / LLM history compression
        
        # Action space: each agent chooses integer harvest 0 to max_harvest
        self.action_space = list(range(max_harvest + 1))
    
    def reset(self, seed: int = None) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.S = self.rng.uniform(*self.initial_S_range)
        self.step_count = 0
        self.history = []
        
        return self._get_observation()
    
    def step(self, actions: List[int]) -> Tuple[Dict, List[float], bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            actions: list of harvest amounts, one per agent (integers 0..max_harvest)
        
        Returns:
            obs, rewards, done, info
        """
        assert len(actions) == self.n_agents, "Must provide one action per agent"
        
        # Clip actions to valid range
        actions = [max(0, min(int(a), self.max_harvest)) for a in actions]
        total_harvest = sum(actions)
        
        # === Stochastic dynamics (core of the environment) ===
        # Logistic growth on current stock before harvest effect
        growth = self.r * self.S * (1 - self.S / self.K)
        
        # Stochastic noise (can be positive or negative shocks)
        noise = self.rng.normal(0, self.noise_std)
        
        # Update stock: growth - harvesting + noise, clipped
        new_S = self.S + growth - total_harvest + noise
        self.S = float(np.clip(new_S, 0.0, self.S_max))
        
        # Rewards: simple individual harvest (can be extended with shared terms)
        rewards = [float(a) for a in actions]
        
        # Record for history / LLM prompting
        self.history.append({
            "step": self.step_count,
            "S_before": round(self.S + total_harvest - growth - noise, 2),  # approximate
            "S_after": round(self.S, 2),
            "actions": actions,
            "total_harvest": total_harvest,
            "growth": round(growth, 2),
            "noise": round(noise, 2),
            "rewards": [round(r, 2) for r in rewards]
        })
        
        self.step_count += 1
        
        # Termination conditions
        done = (self.step_count >= self.max_steps) or (self.S <= 0.1)
        
        obs = self._get_observation()
        info = {
            "stock": round(self.S, 2),
            "total_harvest_this_step": total_harvest,
            "step": self.step_count
        }
        
        return obs, rewards, done, info
    
    def _get_observation(self) -> Dict[str, Any]:
        """Return current observation (can be extended for partial observability)."""
        return {
            "current_stock": round(self.S, 2),
            "step": self.step_count,
            "max_stock": self.S_max,
            "n_agents": self.n_agents
        }
    
    def get_full_history(self) -> List[Dict]:
        """Return the full interaction history (useful for shaper prompts)."""
        return self.history.copy()
    
    def get_compressed_history(self, last_n: int = 10) -> str:
        """
        Simple rule-based compression for LLM prompts.
        You can replace this with an LLM summarizer later for richer compression.
        """
        if not self.history:
            return "No history yet."
        
        recent = self.history[-last_n:]
        
        lines = []
        for h in recent:
            line = (f"Step {h['step']}: Stock after = {h['S_after']}, "
                    f"Actions = {h['actions']}, Total harvested = {h['total_harvest']}, "
                    f"Growth = {h['growth']}, Noise = {h['noise']}")
            lines.append(line)
        
        avg_harvest = np.mean([h['total_harvest'] for h in recent])
        avg_stock = np.mean([h['S_after'] for h in recent])
        
        summary = "\n".join(lines)
        summary += f"\n\nRecent average total harvest per step: {avg_harvest:.1f}"
        summary += f"\nRecent average stock level: {avg_stock:.1f}"
        
        return summary
    
    def render(self):
        """Simple text render for debugging."""
        print(f"Step {self.step_count} | Stock: {self.S:.1f} | "
              f"Last actions: {self.history[-1]['actions'] if self.history else 'N/A'}")


# ====================== Quick test ======================
if __name__ == "__main__":
    env = StochasticCPR(n_agents=2, max_steps=30, seed=42)
    obs = env.reset()
    print("Initial obs:", obs)
    
    for t in range(5):
        actions = [env.rng.integers(0, 5), env.rng.integers(0, 5)]
        obs, rewards, done, info = env.step(actions)
        print(f"Actions: {actions} -> Rewards: {rewards} | Stock: {info['stock']}")
        if done:
            break
    
    print("\nCompressed history sample:")
    print(env.get_compressed_history(last_n=5))
