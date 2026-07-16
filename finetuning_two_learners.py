import argparse
import torch 

from agents import AgentConfig, PPOAgent
from environment import GameParams, IteratedMatrixGame, outer_rollout
from observation_managers import ObservationManagerConfig
from utils.device_utils import empty_device_cache, get_device_str
from utils.training_utils import set_seed
from utils.file_management_utils import load_from_json, save_to_json, validate_config

def main(): 

    print("Packages imported")
    print(f"Using device: {get_device_str()}")
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to the configuration file')
    parser.add_argument('saving_path', type=str, help='Saving path')
    parser.add_argument('--n_seeds', type=int, default=3, help='Number of seeds')
    parser.add_argument('--no_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--checkpoint_freq', type=int, default=50, help='Save a model checkpoint every N epochs. If not set, no checkpoints are saved.')

    args = parser.parse_args()
    full_config = load_from_json(args.config_path)
    validate_config(full_config, "two_learners")
    saving_path = args.saving_path
    n_seeds, no_epochs, checkpoint_freq = args.n_seeds, args.no_epochs, args.checkpoint_freq

    print("Arguments parsed")

    # Instantiate configs
    game_params = GameParams(**full_config["game_parameters"])
    obs_manager_config1 = ObservationManagerConfig(**full_config["obs_manager_parameters1"])
    obs_manager_config2 = ObservationManagerConfig(**full_config["obs_manager_parameters2"])
    ppo_agent_config1 = AgentConfig(**full_config["ppo_agent_parameters1"])
    ppo_agent_config2 = AgentConfig(**full_config["ppo_agent_parameters2"])

    # Generate n_seeds different seeds for the experiment 
    seeds, exp_ids = list(range(n_seeds)), list(range(1, n_seeds+1))
    experiment_path = lambda x: f"{saving_path}/exp{x}_"

    print("Ready to start experiments")
    
    for seed, ind in zip(seeds, exp_ids):

        seed = int(seed)
        set_seed(seed)

        # Initialise agents 
        agent1 = PPOAgent(ppo_agent_config1)
        agent2 = PPOAgent(ppo_agent_config2)
        print("Agents initialised.")

        # Initialise game
        game = IteratedMatrixGame(game_params, obs_manager_config1, obs_manager_config2)


        for epoch in range(no_epochs):

            print(f"Starting epoch {epoch+1}:")
            traj_data1, traj_data2, outcomes = outer_rollout(game, agent1, agent2)

            for model_id, (agent, data) in enumerate(zip([agent1, agent2], [traj_data1, traj_data2])):
                if agent.is_shaper:
                    agent.update_parameters(data) # update agent parameters
                    agent.update_vf_coef() # Update value function coefficient

            if checkpoint_freq and (((epoch+1) % checkpoint_freq) == 0 or epoch == (no_epochs - 1)):
                agent1.trainer.save_pretrained(experiment_path(ind) + f"model1_model_checkpoint_{epoch+1}")
                agent2.trainer.save_pretrained(experiment_path(ind) + f"model2_model_checkpoint_{epoch+1}")

        agent1.logger.save_stats(experiment_path(ind) + "model1_")
        agent2.logger.save_stats(experiment_path(ind) + "model2_")
        save_to_json(game.outcomes, experiment_path(ind) + "all_round_outcomes")

        del agent1, agent2
        empty_device_cache()
        print(f"Experiment {ind} completed.")


if __name__ == "__main__":
    main()
