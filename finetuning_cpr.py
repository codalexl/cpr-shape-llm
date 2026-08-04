"""
Training entry point for the deterministic CPR game.

Structurally a copy of finetuning_two_learners.py, kept separate so the RPS/IPD entry
point stays frozen while the CPR path stabilises. The PPO/LoRA machinery underneath
(outer_rollout, PPOAgent, StatsLogger) is reused unchanged.

The one substantive addition is serializing `game.records`. finetuning_two_learners.py
saves only the logger stats and the outcome indices, so nothing in the reused machinery
writes per-step resource levels, requests or receipts. Every CPR metric — survival curves,
horizon-30 truncated return, the conditionality table — is computed from records, so if
this file does not write them they are silently lost.
"""

import argparse

from agents import AgentConfig, PPOAgent
from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import CPRObservationManagerConfig
from environment import outer_rollout
from utils.device_utils import empty_device_cache, get_device_str
from utils.file_management_utils import load_from_json, save_to_json, validate_config
from utils.training_utils import set_seed


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
    validate_config(full_config, "cpr_two_learners")
    saving_path = args.saving_path
    n_seeds, no_epochs, checkpoint_freq = args.n_seeds, args.no_epochs, args.checkpoint_freq

    print("Arguments parsed")

    # Instantiate configs
    game_params = CPRGameParams(**full_config["game_parameters"])
    obs_manager_config1 = CPRObservationManagerConfig(**full_config["obs_manager_parameters1"])
    obs_manager_config2 = CPRObservationManagerConfig(**full_config["obs_manager_parameters2"])
    ppo_agent_config1 = AgentConfig(**full_config["ppo_agent_parameters1"])
    ppo_agent_config2 = AgentConfig(**full_config["ppo_agent_parameters2"])

    # Generate n_seeds different seeds for the experiment
    seeds, exp_ids = list(range(n_seeds)), list(range(1, n_seeds + 1))
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
        game = CPRGame(game_params, obs_manager_config1, obs_manager_config2)

        for epoch in range(no_epochs):

            print(f"Starting epoch {epoch+1}:")
            game.epoch = epoch  # stamped onto every record row for later alignment
            traj_data1, traj_data2, outcomes = outer_rollout(game, agent1, agent2)

            for agent, data in zip([agent1, agent2], [traj_data1, traj_data2]):
                if agent.is_shaper:
                    agent.update_parameters(data)  # shapers update once per trial
                    agent.update_vf_coef()

            _print_epoch_summary(game, epoch)

            if checkpoint_freq and (((epoch + 1) % checkpoint_freq) == 0 or epoch == (no_epochs - 1)):
                agent1.trainer.save_pretrained(experiment_path(ind) + f"model1_model_checkpoint_{epoch+1}")
                agent2.trainer.save_pretrained(experiment_path(ind) + f"model2_model_checkpoint_{epoch+1}")

        agent1.logger.save_stats(experiment_path(ind) + "model1_")
        agent2.logger.save_stats(experiment_path(ind) + "model2_")
        save_to_json(game.outcomes, experiment_path(ind) + "all_round_outcomes")
        save_to_json(game.records, experiment_path(ind) + "cpr_records")
        print(f"Per-step records saved to {experiment_path(ind)}cpr_records")

        del agent1, agent2
        empty_device_cache()
        print(f"Experiment {ind} completed.")


def _print_epoch_summary(game: CPRGame, epoch: int) -> None:
    """Per-epoch CPR health check: survival, returns, and the masked fraction.

    The masked fraction is the diagnostic for §2.2's cost model — it is also the share of
    the trial that carries no learning signal, so a run stuck near 1.0 is collapsing
    immediately every episode."""
    records = game.records
    rows_this_epoch = [i for i, e in enumerate(records["epoch"]) if e == epoch]
    if not rows_this_epoch:
        return

    masked = [records["masked"][i] for i in rows_this_epoch]
    reward_1 = sum(records["reward_1"][i] for i in rows_this_epoch)
    reward_2 = sum(records["reward_2"][i] for i in rows_this_epoch)
    survived = sum(1 for i in rows_this_epoch
                   if records["step"][i] == game.t_max and not records["depleted"][i])
    episodes = game.e_max * game.n_games

    print(f"\nEpoch {epoch+1} CPR summary — "
          f"survived {survived}/{episodes} episodes | "
          f"returns {reward_1/game.n_games:.1f} / {reward_2/game.n_games:.1f} per game | "
          f"masked {sum(masked)/len(masked):.0%} of steps")


if __name__ == "__main__":
    main()
