import json 
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Optional

def merge_dicts(dict_list:Dict): 

   merged_dict = {key: [item for dic in dict_list for item in dic[key]] for key in list(dict_list[0].keys())}
   return merged_dict

def save_to_json(dictionary: Dict, file_path:str) -> None:
    with open(file_path, 'w') as f: 
        json.dump(dictionary, f, indent=2)


def load_from_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_txt(file_path:str, file_content:str):
    with open(file_path + '.txt', 'w') as file:
        file.write(file_content)


def read_txt(file_path:str) -> str:
    with open(file_path + '.txt', 'r') as file:
        file_content = ' '.join(file.readlines())
    return file_content


def save_dict_as_txt(file_path:str, dictionary:Dict):
    with open(file_path + '.txt', 'w') as file:
        file.write(json.dumps(dictionary))


def validate_config(config: Dict, simulation_type:str = "fixed_opponent") -> None:
    """Function to validate that the parameters in the configuration file are consistent"""
    if simulation_type == "fixed_opponent": 
        assert config["game_parameters"]["r_matrix"] == config["obs_manager_parameters"]["reward_matrix"], "The reward matrices in the game parameters and observation manager must be the same"
        assert config["obs_manager_parameters"]["is_shaper"] == config["ppo_agent_parameters"]["is_shaper"], "The is_shaper parameter must have the same value in the observation manager and agent configs"
        assert config["fixed_agent_parameters"]["allowed_tokens"]["a1_tok"] == config["obs_manager_parameters"]["a1_tok"], "The allowed token for action 1 must be the same in the fixed agent parameters and observation manager"
        assert config["fixed_agent_parameters"]["allowed_tokens"]["a2_tok"] == config["obs_manager_parameters"]["a2_tok"], "The allowed token for action 2 must be the same in the fixed agent parameters and observation manager"
        assert config["fixed_agent_parameters"]["n_games"] == config["game_parameters"]["n_games"], "The number of games must be the same with the fixed agent parameters and the game parameters"

    elif simulation_type == "two_learners":
        assert config["game_parameters"]["r_matrix"] == config["obs_manager_parameters1"]["reward_matrix"], "The reward matrices in the game parameters and observation manager 1 must be the same"
        assert [config["game_parameters"]["r_matrix"][-1], config["game_parameters"]["r_matrix"][0]] == config["obs_manager_parameters2"]["reward_matrix"], "The reward matrices in the game parameters and observation manager 2 must be the same"
        assert config["obs_manager_parameters1"]["is_shaper"] == config["ppo_agent_parameters1"]["is_shaper"], "The is_shaper parameter must have the same value in observation manager 1 and agent config 1"
        assert config["obs_manager_parameters2"]["is_shaper"] == config["ppo_agent_parameters2"]["is_shaper"], "The is_shaper parameter must have the same value in observation manager 2 and agent config 2"

    elif simulation_type == "eval_two_learners":
        assert config["game_parameters"]["r_matrix"] == config["obs_manager_parameters1"]["reward_matrix"], "The reward matrices in the game parameters and observation manager 1 must be the same"
        assert [config["game_parameters"]["r_matrix"][-1], config["game_parameters"]["r_matrix"][0]] == config["obs_manager_parameters2"]["reward_matrix"], "The reward matrices in the game parameters and observation manager 2 must be the same"
        assert config["obs_manager_parameters1"]["is_shaper"] == config["agent_parameters1"]["is_shaper"], "The is_shaper parameter must have the same value in observation manager 1 and agent config 1"
        assert config["obs_manager_parameters2"]["is_shaper"] == config["agent_parameters2"]["is_shaper"], "The is_shaper parameter must have the same value in observation manager 2 and agent config 2"

@dataclass
class StatsLogger():
    """Holds training statics for a given model"""
    total_loss: List = field(default_factory=list)
    policy_loss: List = field(default_factory=list)
    value_loss: List = field(default_factory=list)
    obj_entropy: List = field(default_factory=list)
    entropy: List = field(default_factory=list)
    mean_scores: List = field(default_factory=list)
    std_score: List = field(default_factory=list)
    kl_divergences: List = field(default_factory=list)
    policy_ratio: List = field(default_factory=list)
    kl_coefs: List = field(default_factory=list)
    grad_keys: List[str] = field(default_factory=lambda: ["lora_A", "lora_B", "v_head_weight", "v_head_bias"])
    track_gradients: Optional[bool] = False
    
    
    def __post_init__(self):
        if self.track_gradients: 
            self.avg_grad_norm = {key: [] for key in self.grad_keys}
            self.std_grad_norm = {key: [] for key in self.grad_keys}

    def log_stats(self, stats:Dict): 
        """Appends training statistics to the relevant attribute"""
        self.total_loss.append(stats["ppo/loss/total"])
        self.policy_loss.append(stats["ppo/loss/policy"])
        self.value_loss.append(stats["ppo/loss/value"])
        self.entropy.append(stats["ppo/policy/entropy"])
        self.mean_scores.append(stats["ppo/mean_scores"])
        self.std_score.append(stats["ppo/std_scores"])
        self.kl_divergences.append(stats["objective/kl"])
        self.policy_ratio.append(np.mean(stats["ppo/policy/ratio"], dtype=np.float64))
        self.obj_entropy.append(stats["objective/entropy"])
        self.kl_coefs.append(stats["objective/kl_coef"])

        if self.track_gradients:
            for key in self.grad_keys:
                self.avg_grad_norm[key].append(np.mean(stats["temp_grads"][key]))
                self.std_grad_norm[key].append(np.std(stats["temp_grads"][key]))
        
    
    def get_stats_dict(self) -> Dict:
        """Constructs dictionary with relevant statistics"""
        training_metrics_dict = {"total_loss": self.total_loss, "policy_loss": self.policy_loss,
                          "value_loss": self.value_loss, "entropy": self.entropy, "mean_scores": self.mean_scores,
                          "std_score": self.std_score, "kl_div": self.kl_divergences, "obj_entropy": self.obj_entropy,
                          "policy_ratio": self.policy_ratio, "kl_coef": self.kl_coefs}
        
        if self.track_gradients: 
            for key in self.grad_keys:
                training_metrics_dict[f"grad_avg_"+ key] =  self.avg_grad_norm[key]
                training_metrics_dict[f"grad_std_"+ key] =  self.std_grad_norm[key]
                
        return training_metrics_dict 
    
    def save_stats(self, saving_path: str) -> None:
        """Saves the training metrics and paramters dictionary as a txt"""
        training_metrics_dict = self.get_stats_dict()
        save_dict_as_txt(saving_path + "training_metrics", training_metrics_dict)
        print("The parameters and training statistics have been successfully saved.")

    def append_stats(self, saving_path: str) -> None: 
        """Recovers training statistics saved from previous round, and adds the current training statistics to the file. This 
        function is called whenever we want to continue training from the saved checkpoints of a previous experiment. Note that
        if in the previous experiment there was NO gradient tracking, they will not be tracked even if the track_gradient arguments
        specifies otherwise. This should not be a problem as the configuration file for the continuation run should be the same as 
        that of the original run."""
        # Retrieve original training statistics
        original_training_stats = json.loads(read_txt(saving_path + "training_metrics"))
        new_training_stats = self.get_stats_dict()
        merged_training_stats = merge_dicts([original_training_stats, new_training_stats])
        save_dict_as_txt(saving_path + "training_metrics", merged_training_stats)
        print(f"The training statistics have been successfully updated in path {saving_path + 'training_metrics'}.")