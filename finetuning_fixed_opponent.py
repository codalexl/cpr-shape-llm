import argparse
import torch

from agents import AgentConfig, FixedAgent, FixedAgentConfig, PPOAgent
from environment import EnvState, GameParams, inner_rollout_fixed_opponent, IteratedMatrixGame, TrajectoryData 
from observation_managers import ObservationManagerConfig
from utils.file_management_utils import load_from_json, save_to_json, validate_config
from utils.device_utils import empty_device_cache, get_device_str
from utils.training_utils import set_seed

def main(): 

    print("Packages imported")
    print(f"Using device: {get_device_str()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('saving_path', type=str, help='Saving path')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--no_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Save a model checkpoint every N epochs. If not set, no checkpoints are saved.')

    args = parser.parse_args()
    full_config = load_from_json(args.config_path)
    validate_config(full_config, "fixed_opponent")
    saving_path = args.saving_path
    n_seeds, no_epochs, checkpoint_freq = args.n_seeds, args.no_epochs, args.checkpoint_freq

    print("Arguments parsed")

    # Instantiate configs
    game_params = GameParams(**full_config["game_parameters"])
    obs_manager_config = ObservationManagerConfig(**full_config["obs_manager_parameters"])
    ppo_agent_config = AgentConfig(**full_config["ppo_agent_parameters"])
    fixed_agent_config = FixedAgentConfig(**full_config["fixed_agent_parameters"])

    # Generate n_seeds different seeds for the experiment 
    seeds, exp_ids = list(range(n_seeds)), list(range(1, n_seeds+1))
    experiment_path = lambda x: f"{saving_path}/exp{x}_"

    print("Ready to start experiments")
    
    for seed, ind in zip(seeds, exp_ids):

        seed = int(seed)
        set_seed(seed)

        # Initialise agents 
        agent1 = PPOAgent(ppo_agent_config)
        agent2 = FixedAgent(fixed_agent_config)

        # Initialise game
        game = IteratedMatrixGame(game_params, obs_manager_config, obs_manager_config)

        for epoch in range(no_epochs):

            print(f"Starting epoch {epoch+1}:")

            # Initialise environment state and trajectory data
            env_state = EnvState(inner_t=0, outer_t=0)
            traj_data1 = TrajectoryData(last_observation = [game.obs_managers["agent_1"].game_description + game.obs_managers["agent_1"].instruction_prompt] * game.n_games)
            traj_data1, env_state = inner_rollout_fixed_opponent(game, env_state, traj_data1, agent1, agent2) # Rollout full episode
            agent1.update_parameters(traj_data1)
            agent1.update_vf_coef()

            if checkpoint_freq and ((epoch+1) % checkpoint_freq) == 0:
                agent1.trainer.save_pretrained(experiment_path(ind) + f"_model_checkpoint_{epoch+1}")

        agent1.logger.save_stats(experiment_path(ind))
        save_to_json(game.outcomes, experiment_path(ind) + "all_round_outcomes")

        del agent1
        empty_device_cache()
        print(f"Experiment {ind} completed.")

if __name__ == "__main__":
    main()

    


