import os 
import itertools 
import numpy as np
import torch

from dataclasses import dataclass, field
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOConfig, AutoModelForCausalLMWithValueHead
from typing import List, Dict, Optional

from utils.dataset_utils import simple_collator
from utils.file_management_utils import StatsLogger
from utils.training_utils import CustomPPOTrainer


@dataclass 
class FixedAgentConfig():
    """Holds parameters to initialise a fixed agent"""
    agent_id: int
    agent_type: str
    allowed_tokens: Dict
    n_games: int
    n_tats: Optional[int] = 1

    def __post_init__(self):
        """Check the agent type selected is available"""
        assert self.agent_type in ["random", "TFT"], f"The only available fixed agents are random and Tit For N Tats; received {self.agent_type}" 

@dataclass
class AgentConfig():
    """Holds paramters for LLM-based agent trained via PPO"""
    agent_id: int
    is_shaper: bool
    model_path: str
    adapter_path: str
    ppo_params: Dict
    generation_kwargs: Dict = field(default_factory=lambda: {"min_length": -1, "max_new_tokens": 1, "top_k": 0.0, "top_p": 1.0, "do_sample": True})
    vf_coef_adapt_params: Optional[Dict] = None
    init_entropy_coef: Optional[float] = 0.
    final_entropy_coef: Optional[float] = 0.
    entropy_coef_horizon: Optional[int] = 1000
    print_training_stats: Optional[bool] = True
    track_gradients: Optional[bool] = False
    training_cont: Optional[bool] = False
    a1_tok: Optional[int] = 235288
    a2_tok: Optional[int] = 235299

@dataclass
class EvalAgentConfig():
    """Holds paramters for LLM-based agent for inference only"""
    agent_id: int
    is_shaper: bool
    model_path: str
    adapter_path: str
    generation_kwargs: Dict = field(default_factory=lambda: {"min_length": -1, "max_new_tokens": 1, "top_k": 0.0, "top_p": 1.0, "do_sample": True})


class PPOAgent():
    """LLM based agent trained via PPO - can use adapters for training"""
    def __init__(self, config: AgentConfig): 

        # Unpack elements from config
        self.agent_id, self.is_shaper = config.agent_id, config.is_shaper
        self.ppo_params, self.generation_kwargs =  config.ppo_params, config.generation_kwargs
        self.vf_coef_adapt_params = config.vf_coef_adapt_params
        self.print_training_stats = config.print_training_stats
        self.vf_coef = config.ppo_params["vf_coef"]

        # Initialise tokenizer and load model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        if "llama" in config.model_path: 
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
        if "qwen" in config.model_path:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=torch.bfloat16, attn_implementation='eager')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_path, attn_implementation='eager') 

        # Load adapter if any provided
        is_peft_model, peft_config = False, None
        if config.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path, is_trainable=True)
            is_peft_model, peft_config = True, self.model.peft_config 
        
        # Add Value Head Wrapper
        self.model = AutoModelForCausalLMWithValueHead(self.model)
        self.model.is_peft_model, self.model.peft_config = is_peft_model, peft_config # Necessary as we are loading adapter from file

        if config.training_cont: 
            value_head_state = torch.load(os.path.join(config.adapter_path, "pytorch_model.bin"), map_location='cpu')
            self.model.load_state_dict(value_head_state, strict=False)  # strict=False to only load value head

        # Initialise PPO Trainer
        self.trainer_config = PPOConfig(model_name = config.model_path, **config.ppo_params)
        self.trainer = CustomPPOTrainer(init_entropy_coef = config.init_entropy_coef, final_entropy_coef = config.final_entropy_coef, entropy_coef_horizon = config.entropy_coef_horizon, \
                 a1_tok=config.a1_tok, a2_tok=config.a2_tok, track_gradients = config.track_gradients, model=self.model, config=self.trainer_config, dataset=None, tokenizer=self.tokenizer, data_collator=simple_collator)

        # Initialise stats logger
        self.logger = StatsLogger(track_gradients = config.track_gradients)

    def tokenize_observation(self, obs: List[str]) -> List[torch.Tensor]:
        """Tokenize observations. returns a List of torch.Tensors as it is the format required by the PPO Trainer"""
        input_ids = [self.tokenizer.encode(new_sent, return_tensors="pt").squeeze() for new_sent in obs]
        return input_ids
    
    def take_action(self, query_tensors: List[int]) -> List[torch.Tensor]: 
        """Generate model response. Ensure it is restricted to one token."""
        response_tensors = self.trainer.generate(query_tensors, return_prompt = False, **self.generation_kwargs)
        
        # Ensure the model only replied with one token
        for response in response_tensors:
            assert response.shape == (1, )

        return response_tensors
    
    def update_parameters(self, traj_data) -> None: 
        """Updates the parameters of the model based on the input trajectory."""
        # "Flatten" all queries, responses, and rewards
        query_tensors = list(itertools.chain(*traj_data.query_tensors)) 
        response_tensors = list(itertools.chain(*traj_data.response_tensors)) 
        rewards = list(itertools.chain(*traj_data.rewards)) 
        
        self.trainer.config.batch_size = len(query_tensors) # Adjust batch size
        self.trainer.env_ids, self.trainer.n = list(itertools.chain(*traj_data.env_ids)), len(set(itertools.chain(*traj_data.env_ids))) # Update environment ids for multi-turn advantage calculation

        stats = self.trainer.step(query_tensors, response_tensors, rewards) # Update parameters
        if self.trainer.track_gradients:
            stats["temp_grads"] = self.trainer.gradient_tracker.temp_grad_norm
        self._print_stats(stats) # Print key statistics 
        self.logger.log_stats(stats)

        self.trainer.update_entropy_coef()# Update entropy coefficient
    
    def update_vf_coef(self) -> None:
        """Update the value function coefficient in the total PPO loss."""
        if self.vf_coef_adapt_params:
            policy_loss, value_loss = self.logger.policy_loss[-1], self.logger.value_loss[-1] #recover policy and value loss
            new_vf_coef = self.vf_coef + self.vf_coef_adapt_params["alpha"] * ((abs(policy_loss)/(value_loss*self.vf_coef_adapt_params["target"])) - self.vf_coef)
            max_value = self.vf_coef_adapt_params.get("max_value", 10.0)
            self.vf_coef = np.clip(new_vf_coef, 0, max_value)
            self.trainer.config.vf_coef = self.vf_coef
         

    def _print_stats(self, stats): 
        """Print the key learning statistics for a given parameter update."""
        if self.print_training_stats:
            print(f"\nParameters of model {self.agent_id} updated.")
            print(f'\nTotal loss: {stats["ppo/loss/total"]:.3f}. \nMean Score: {stats["ppo/mean_scores"]:.3f}+/-{stats["ppo/std_scores"]:.3f}')
            print(f'\nValue loss: {stats["ppo/loss/value"]:.3f}.\nPolicy loss: {stats["ppo/loss/policy"]:.3f}.')
            print(f'\nEntropy: {stats["ppo/policy/entropy"]:.3f}. \nPolicy Ratio: {np.mean(stats["ppo/policy/ratio"]):.3f}+/-{np.std(stats["ppo/policy/ratio"]):.3f}.\nKL-Div: {stats["objective/kl"]:.3f}')


class FixedAgent():
    """Fixed Opponent. Returns actions in the same format as LLM-based agent."""
    def __init__(self, config:FixedAgentConfig):
        
        self.agent_id = config.agent_id
        self.agent_type, self.n_tats = config.agent_type, config.n_tats
        self.a1_tok, self.a2_tok = config.allowed_tokens["a1_tok"], config.allowed_tokens["a2_tok"]
        self.n_games = config.n_games
    
    def _get_random_actions(self) -> List[torch.Tensor]:
        """Generates actions sampled from a random policy"""
        action_list = np.random.choice([self.a1_tok, self.a2_tok], (self.n_games)).tolist()
        tensor_list = list(torch.tensor(action_list))
        return tensor_list
    
    def _get_tfnt_actions(self, opp_actions:torch.Tensor) -> List[torch.Tensor]: 
        """Generates actions sampled from a TFNT policy."""
        # Initialise the actions to be all cooperative 
        actions = torch.full((self.n_games,), self.a1_tok)

        if len(opp_actions) >= self.n_tats: # TFNT cooperates the first N times
            n_action_sum = opp_actions[-self.n_tats:].sum(axis=0)
            defection_ids = (n_action_sum == self.n_tats) # Only defect if opponent defected in the last n moves
            actions[defection_ids] = self.a2_tok
        
        return list(actions.reshape(-1, 1))
    
    def take_action(self, params:Dict)-> List[torch.Tensor]:
        """Generates actions throughout gameplay, calling the appropariate generation function for each agent"""
        actions = self._get_random_actions() if self.agent_type == "random" else self._get_tfnt_actions(**params)
        return actions

class EvaluationAgent():
    """LLM based agent used for INFERENCE ONLY"""
    def __init__(self, config: EvalAgentConfig): 

        # Unpack elements from config
        self.agent_id, self.is_shaper = config.agent_id, config.is_shaper
        self.generation_kwargs =  config.generation_kwargs

        # Initialise tokenizer and load model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(config.model_path, attn_implementation='eager') 

        # Load adapter if any provided
        if config.adapter_path:
            self.model = PeftModel.from_pretrained(self.model, config.adapter_path, is_trainable=True)
        
        # Put model in evaluation mode and move to GPU
        self.model.eval()
        self.model = self.model.to("cuda")
    
    def extract_probabilities(self, prompts:List[str], a1_strs:List[str], a2_strs:List[str], device:str="cuda"): 

        assert len(prompts) == len(a1_strs) == len(a2_strs), "The observation and legal tokens lists have different lengths"

        # Get token IDs of the legal strings
        a1_tokens = self.tokenizer(a1_strs, padding=True, return_tensors="pt")["input_ids"][:, -1].tolist()
        a2_tokens = self.tokenizer(a2_strs, padding=True, return_tensors="pt")["input_ids"][:, -1].tolist()

        padded_inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        result = self.model(**padded_inputs) # Forward pass 
        probs = torch.softmax(result["logits"], axis =-1) # Convert logits to probabilities

        a1_probs = probs[list(range(len(prompts))), -1, a1_tokens].tolist() 
        a2_probs = probs[list(range(len(prompts))), -1, a2_tokens].tolist() 

        # Create df with relevant probabilities for storage
        return (a1_probs, a2_probs)
    
    def tokenize_observation(self, obs: List[str]) -> torch.Tensor:
        """Tokenize observations. returns a List of torch.Tensors as it is the format required by the PPO Trainer"""
        input_ids = self.tokenizer(obs, padding=True, return_tensors="pt")["input_ids"].to("cuda")
        return input_ids

    def take_action(self, query_tensors: torch.Tensor) -> torch.Tensor: 
        """Generate model response. Ensure it is restricted to one token."""
        response_tensors = self.model.generate(query_tensors, **self.generation_kwargs).to("cpu")
        return list(response_tensors[:, -1])

