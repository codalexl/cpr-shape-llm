import argparse
import torch 

from agents import EvalAgentConfig, EvaluationAgent
from environment import GameParams, IteratedMatrixGame, outer_rollout
from observation_managers import ObservationManagerConfig
from utils.evaluation_utils import game_play
from utils.training_utils import set_seed
from utils.file_management_utils import load_from_json, save_to_json, validate_config

def main(): 

    print("Packages imported")
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('saving_path', type=str, help='Saving path')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    args = parser.parse_args()
    full_config = load_from_json(args.config_path)
    validate_config(full_config, "eval_two_learners")
    saving_path = args.saving_path
    print("Arguments parsed")

    # Instantiate configs
    game_params = GameParams(**full_config["game_parameters"])
    obs_manager_config1 = ObservationManagerConfig(**full_config["obs_manager_parameters1"])
    obs_manager_config2 = ObservationManagerConfig(**full_config["obs_manager_parameters2"])
    agent_config1 = EvalAgentConfig(**full_config["agent_parameters1"])
    agent_config2 = EvalAgentConfig(**full_config["agent_parameters2"])
    print("Ready to start experiments")

    set_seed(args.seed)

    # Initialise agents
    agent1 = EvaluationAgent(agent_config1)
    agent2 = EvaluationAgent(agent_config2)
    print("Agents initialised.")

    # Initialise game
    game = IteratedMatrixGame(game_params, obs_manager_config1, obs_manager_config2)
    game_play(game, agent1, agent2) # Play game
    save_to_json(game.outcomes, saving_path + "gameplay_outcomes")
    print("Finished Evaluation")


if __name__ == "__main__":
    main()
