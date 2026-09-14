"""
One PPO learner against a non-learning partner: the pond's Tests A/B, and the gate checks and evaluation runs of
the two-player stochastic CPR (docs/PREREGISTRATION_STOCHASTIC_CPR.md).

Reuses outer_rollout unchanged. Only the learner trains; records still include both columns of requests.
The partner is, in order of precedence:
  * a frozen adapter (`--partner_adapter` / `frozen_partner_adapter`): the transfer arm, E1, E3 and E4;
  * a scripted partner (`scripted_partner` in the config): take, tit_for_tat, probe or replay
    (cpr_bots.make_scripted_partner), reading the game's structured state rather than the prompt;
  * a constant action (`fixed_partner_action`, a token index): the pond's Test A (2) and Test B (1).
With `--learner_adapter` / `frozen_learner_adapter` the learner is frozen as well: an evaluation run with no
updates and no checkpoints (E3, E4 and the probes).
"""

import argparse

from agents import AgentConfig, PPOAgent
from cpr_bots import ConstantActionAgent, FrozenAdapterAgent, make_scripted_partner
from cpr_game import CPRGame, CPRGameParams
from cpr_observation_managers import CPRObservationManagerConfig
from environment import outer_rollout
from finetuning_cpr import _print_epoch_summary, _print_live_a_raw, _print_opening_a0
from utils.device_utils import empty_device_cache, get_device_str
from utils.file_management_utils import load_from_json, save_to_json
from utils.training_utils import set_seed


def _validate_learner(config: dict) -> None:
    gp = config["game_parameters"]
    obs1, obs2 = config["obs_manager_parameters1"], config["obs_manager_parameters2"]
    assert obs1["R0"] == obs2["R0"] == gp["R0"]
    assert obs1["is_shaper"] is False and config["ppo_agent_parameters1"]["is_shaper"] is False
    assert len(obs1["action_toks"]) == int(gp["n_actions"])


def _validate(config: dict) -> int:
    action = int(config["fixed_partner_action"])
    n = int(config["game_parameters"]["n_actions"])
    assert 0 <= action < n, f"fixed_partner_action {action} not in 0..{n - 1}"
    _validate_learner(config)
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
    parser.add_argument("--learner_adapter", type=str, default=None,
                        help="Evaluation run: agent 1 is this saved adapter, frozen; no updates and no checkpoints. "
                             "Overrides config key frozen_learner_adapter.")
    parser.add_argument("--replay_records", type=str, default=None,
                        help="Replay partner (E2): the recorded run whose takes are replayed. "
                             "Overrides scripted_partner.records.")
    parser.add_argument("--replay_window", type=int, default=None,
                        help="Smoke runs only: replay this many recorded epochs instead of the pre-registered 20.")
    args = parser.parse_args()

    full_config = load_from_json(args.config_path)
    partner_adapter = args.partner_adapter or full_config.get("frozen_partner_adapter")
    learner_adapter = args.learner_adapter or full_config.get("frozen_learner_adapter")
    scripted = full_config.get("scripted_partner")
    partner_cfg, partner_action = None, -1
    if partner_adapter:
        assert "ppo_agent_parameters2" in full_config, "a frozen partner needs ppo_agent_parameters2"
        p2 = dict(full_config["ppo_agent_parameters2"]); p2["adapter_path"] = partner_adapter
        partner_cfg = AgentConfig(**p2)
        _validate_learner(full_config)
        print(f"Transfer arm: learner vs frozen adapter {partner_adapter}")
    elif scripted:
        scripted = dict(scripted)
        if scripted.get("kind") == "replay" and args.replay_records:
            scripted["records"] = args.replay_records
        if scripted.get("kind") == "replay" and args.replay_window:
            scripted["window"] = args.replay_window
        _validate_learner(full_config)
        print(f"Scripted partner: {scripted.get('kind')} {({k: v for k, v in scripted.items() if k != 'kind'})}")
    else:
        partner_action = _validate(full_config)
        print(f"Fixed partner: learner vs frozen always-{partner_action}")
    trains = not learner_adapter
    if not trains:
        print(f"Evaluation run: learner frozen at {learner_adapter}; no updates, no checkpoints")

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
        learner = PPOAgent(learner_cfg) if trains else FrozenAdapterAgent(learner_cfg, adapter_path=learner_adapter)
        partner = None
        if partner_adapter:
            partner = FrozenAdapterAgent(partner_cfg, adapter_path=partner_adapter)
            print(f"Frozen adapter partner: {partner_adapter} (is_shaper={partner.is_shaper})")
        elif not scripted:
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
        game.attach_close_table(seed, args.no_epochs, experiment_path(ind) + "close_table.npy")
        if scripted:
            partner = make_scripted_partner(scripted, game, toks, seed=seed, n_epochs=args.no_epochs)

        for epoch in range(args.no_epochs):
            print(f"Starting epoch {epoch + 1}:")
            game.epoch = epoch
            learner.current_epoch = epoch
            traj1, traj2, _ = outer_rollout(game, learner, partner)
            del traj1, traj2
            _print_epoch_summary(game, epoch)
            if trains:
                _print_opening_a0(learner, epoch)
            if epoch == 0:
                # Seed-divergence check reads this before the run finishes (scripts/check_seed_divergence.py).
                _rows = game.records
                _open = [(int(_rows["request_1"][i]), int(_rows["request_2"][i]))
                         for i in range(len(_rows["epoch"])) if int(_rows["epoch"][i]) == 0 and int(_rows["step"][i]) == 1]
                save_to_json({"seed": seed, "openings": _open}, experiment_path(ind) + "epoch1_openings")
            if trains:
                _print_live_a_raw(learner, epoch)
            if trains and args.checkpoint_freq and (
                ((epoch + 1) % args.checkpoint_freq) == 0 or epoch == args.no_epochs - 1
            ):
                learner.trainer.save_pretrained(
                    experiment_path(ind) + f"model1_model_checkpoint_{epoch + 1}"
                )

        save_to_json(game.outcomes, experiment_path(ind) + "all_round_outcomes")
        save_to_json(game.records, experiment_path(ind) + "cpr_records")
        print(f"Per-step records saved to {experiment_path(ind)}cpr_records")
        if trains:
            learner.logger.save_stats(experiment_path(ind) + "model1_")
            save_to_json(learner.opening_log, experiment_path(ind) + "opening_a0")
            save_to_json(learner.live_adv_log, experiment_path(ind) + "live_adv")
            print(f"Opening A0 log saved to {experiment_path(ind)}opening_a0  ({len(learner.opening_log)} rows)")
            print(f"Live-step A log saved to {experiment_path(ind)}live_adv  ({len(learner.live_adv_log)} rows)")
        print(f"Partner updates (no-op count): {getattr(partner, 'updates', 'n/a')}")
        del learner, partner
        empty_device_cache()
        print(f"Experiment {ind} completed.")


if __name__ == "__main__":
    main()
