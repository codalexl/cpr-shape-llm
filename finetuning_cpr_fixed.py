"""
CPR Tests A/B: one PPO learner vs a frozen constant-action partner.

Reuses outer_rollout unchanged. The partner is ConstantActionAgent (no Gemma).
Only the learner is trained. Records still include both columns of requests.
`fixed_partner_action` in the config picks the robot's harvest (2 for A, 1 for B).
"""

import argparse

from agents import AgentConfig, PPOAgent
from cpr_bots import ConstantActionAgent
from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import CPRObservationManagerConfig
from environment import outer_rollout
from finetuning_cpr import _print_epoch_summary, _print_live_a_raw, _print_opening_a0
from utils.device_utils import empty_device_cache, get_device_str
from utils.file_management_utils import load_from_json, save_to_json
from utils.training_utils import set_seed


def _validate(config: dict) -> int:
    gp = config["game_parameters"]
    obs1, obs2 = config["obs_manager_parameters1"], config["obs_manager_parameters2"]
    ppo = config["ppo_agent_parameters1"]
    action = int(config["fixed_partner_action"])
    n = int(gp["n_actions"])
    assert 0 <= action < n, f"fixed_partner_action {action} not in 0..{n - 1}"
    assert obs1["R0"] == obs2["R0"] == gp["R0"]
    assert obs1["is_shaper"] is False and ppo["is_shaper"] is False
    assert len(obs1["action_toks"]) == n
    return action


def main():
    print("Packages imported")
    print(f"Using device: {get_device_str()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("saving_path", type=str)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--no_epochs", type=int, default=15)
    parser.add_argument("--checkpoint_freq", type=int, default=0)
    parser.add_argument("--partner_adapter", type=str, default=None,
                        help="Transfer arm: path to a saved adapter; agent 2 is that policy, frozen. "
                             "Overrides config key frozen_partner_adapter.")
    args = parser.parse_args()

    full_config = load_from_json(args.config_path)
    partner_adapter = args.partner_adapter or full_config.get("frozen_partner_adapter")
    partner_cfg = None
    if partner_adapter:
        assert "ppo_agent_parameters2" in full_config, "transfer config needs ppo_agent_parameters2"
        p2 = dict(full_config["ppo_agent_parameters2"]); p2["adapter_path"] = partner_adapter
        partner_cfg = AgentConfig(**p2)
        partner_action = -1
        print(f"Transfer arm: learner vs frozen adapter {partner_adapter}")
    else:
        partner_action = _validate(full_config)
        print(f"Fixed partner: learner vs frozen always-{partner_action}")

    game_params = CPRGameParams(**full_config["game_parameters"])
    obs1 = CPRObservationManagerConfig(**full_config["obs_manager_parameters1"])
    obs2 = CPRObservationManagerConfig(**full_config["obs_manager_parameters2"])
    learner_cfg = AgentConfig(**full_config["ppo_agent_parameters1"])
    toks = list(full_config["obs_manager_parameters2"]["action_toks"])

    experiment_path = lambda x: f"{args.saving_path}/exp{x}_"
    print("Ready to start experiments")

    for seed in range(args.seed_start, args.seed_start + args.n_seeds):
        ind = seed + 1  # exp index is seed + 1 so per-seed launches into one folder never collide
        set_seed(seed)
        print(f"RNG seed {seed} → {experiment_path(ind)}")
        learner = PPOAgent(learner_cfg)
        if partner_adapter:
            from cpr_bots import FrozenAdapterAgent
            partner = FrozenAdapterAgent(partner_cfg, adapter_path=partner_adapter)
            print(f"Frozen adapter partner: {partner_adapter} (is_shaper={partner.is_shaper})")
        else:
            partner = ConstantActionAgent(partner_action, action_toks=toks)
        # TRL 0.11.4 PPOTrainer.__init__ calls transformers.set_seed(config.seed) with the
        # PPOConfig default seed=0, which silently resets the global RNG that the sampler
        # uses. Every experiment before this line was therefore drawn from the seed-0
        # stream (see scripts/audit_seeds_and_scale.py). Re-seed after construction so
        # exp k actually samples from seed k.
        set_seed(seed)
        print("Agents initialised.")
        game = CPRGame(game_params, obs1, obs2)
        game.attach_noise_table(seed, args.no_epochs, experiment_path(ind) + "noise_table.npy")

        for epoch in range(args.no_epochs):
            print(f"Starting epoch {epoch + 1}:")
            game.epoch = epoch
            learner.current_epoch = epoch
            traj1, traj2, _ = outer_rollout(game, learner, partner)
            del traj1, traj2
            _print_epoch_summary(game, epoch)
            _print_opening_a0(learner, epoch)
            if epoch == 0:
                # Seed-divergence check reads this before the run finishes (scripts/check_seed_divergence.py).
                _rows = game.records
                _open = [(int(_rows["request_1"][i]), int(_rows["request_2"][i]))
                         for i in range(len(_rows["epoch"])) if int(_rows["epoch"][i]) == 0 and int(_rows["step"][i]) == 1]
                save_to_json({"seed": seed, "openings": _open}, experiment_path(ind) + "epoch1_openings")
            _print_live_a_raw(learner, epoch)
            if args.checkpoint_freq and (
                ((epoch + 1) % args.checkpoint_freq) == 0 or epoch == args.no_epochs - 1
            ):
                learner.trainer.save_pretrained(
                    experiment_path(ind) + f"model1_model_checkpoint_{epoch + 1}"
                )

        learner.logger.save_stats(experiment_path(ind) + "model1_")
        save_to_json(game.outcomes, experiment_path(ind) + "all_round_outcomes")
        save_to_json(game.records, experiment_path(ind) + "cpr_records")
        save_to_json(learner.opening_log, experiment_path(ind) + "opening_a0")
        save_to_json(learner.live_adv_log, experiment_path(ind) + "live_adv")
        print(f"Per-step records saved to {experiment_path(ind)}cpr_records")
        print(f"Opening A0 log saved to {experiment_path(ind)}opening_a0  ({len(learner.opening_log)} rows)")
        print(f"Live-step A log saved to {experiment_path(ind)}live_adv  ({len(learner.live_adv_log)} rows)")
        print(f"Frozen partner updates (no-op count): {partner.updates}")
        del learner, partner
        empty_device_cache()
        print(f"Experiment {ind} completed.")


if __name__ == "__main__":
    main()
