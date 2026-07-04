from typing import Tuple, Callable, Dict, List
import string
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats

from environment import EnvState, GameParams, IteratedMatrixGame, TrajectoryData, inner_rollout, inner_rollout_fixed_opponent
from agents import EvaluationAgent, EvalAgentConfig, FixedAgent, FixedAgentConfig
from observation_managers import ObservationManagerConfig
from utils.file_management_utils import save_dict_as_txt, save_to_json, load_from_json
from utils.training_utils import set_seed

def extract_all_legal_token_probabilities(query: Callable, agent: EvaluationAgent, batch_size:int=10, saving_path:str=None) -> Tuple[List[float], List[float], List[Tuple[str, str]]]: 
  """Given a query, this function extracts the probability of generating each of the two legal tokens for all possible combinations of two capital letters"""
  
  # Initialise the set of possible legal tokens
  possible_letters = list(itertools.permutations(string.ascii_uppercase, 2))
  a1_strings, a2_strings = zip(*possible_letters)
  a1_strings, a2_strings = list(a1_strings), list(a2_strings)

  all_a1_probs, all_a2_probs = [], []

  # Only required for batching the policy extraction - otherwise run out of RAM
  for i in range(0, len(a1_strings), batch_size):
    queries = [query(x, y) for x,y in zip(a1_strings[i:i+batch_size], a2_strings[i:i+batch_size])]
    a1_probs, a2_probs = agent.extract_probabilities(queries, a1_strings[i:i+batch_size], a2_strings[i:i+batch_size])
    all_a1_probs.extend(a1_probs)
    all_a2_probs.extend(a2_probs)
  
  if saving_path: 
    data = {'prompt': possible_letters, 'a1': all_a1_probs, 'a2': all_a2_probs}
    df = pd.DataFrame(data)
    df.to_csv(f'{saving_path}.csv', index=False) 
    print(f"Evaluation saved in {saving_path}")
  return (all_a1_probs, all_a2_probs, possible_letters)
  

def game_play(game:IteratedMatrixGame, agent1: EvaluationAgent, agent2: EvaluationAgent, fixed_opponent:bool=False, saving_path:str=None) -> Dict[str, List[List[float]]]: 
    """Regular gameplay - with no model training. The outcomes are updated within the game object."""
    # Initialise environment state and trajectories
    env_state = EnvState(inner_t=0, outer_t=0) 
    traj_data1 = TrajectoryData(last_observation = [game.obs_managers["agent_1"].game_description + game.obs_managers["agent_1"].instruction_prompt] * game.n_games)
    traj_data2 = TrajectoryData(last_observation = [game.obs_managers["agent_2"].game_description + game.obs_managers["agent_2"].instruction_prompt] * game.n_games)

    for e in range(game.e_max): 
        if fixed_opponent: 
          traj_data1, env_state = inner_rollout_fixed_opponent(game, env_state, traj_data1, agent1, agent2)
        else: 
          traj_data1, traj_data2, env_state = inner_rollout(game, env_state, traj_data1, traj_data2, agent1, agent2) # rollout trajectory
    
    # Process data
    agent1_rewards = [[] for _ in range(game.n_games)]
    agent2_rewards = [[] for _ in range(game.n_games)]

    for traj, skeleton in zip([traj_data1, traj_data2], [agent1_rewards, agent2_rewards]): 
      for rewards, env_ids in zip(traj.rewards, traj.env_ids):
        for r, id in zip(rewards, env_ids):
          skeleton[id].append(r.item())

    rewards_data = {"agent_1": agent1_rewards, "agent_2": agent2_rewards}
    if saving_path:
      save_dict_as_txt(saving_path, rewards_data)
      print(f"Evaluation data saved in {saving_path}")

    return rewards_data


def evaluate_model_pairs(m1_paths, m2_paths, full_config, saving_tags, seed = 1): 
    
    # Instantiate configs
    game_params = GameParams(**full_config["game_parameters"])
    obs_manager_config1 = ObservationManagerConfig(**full_config["obs_manager_parameters1"])
    obs_manager_config2 = ObservationManagerConfig(**full_config["obs_manager_parameters2"])
    agent_config1 = EvalAgentConfig(**full_config["agent_parameters1"])
    agent_config2 = EvalAgentConfig(**full_config["agent_parameters2"])
    print("Ready to start experiments")

    set_seed(seed)

    for m1, m2, tag in zip(m1_paths, m2_paths, saving_tags):
      # Overwrite adapter configs and initialise agents
      agent_config1.adapter_path, agent_config2.adapter_path = m1, m2
      agent1 = EvaluationAgent(agent_config1)
      agent2 = EvaluationAgent(agent_config2)
      print("Agents initialised.")

      # Initialise game
      game = IteratedMatrixGame(game_params, obs_manager_config1, obs_manager_config2)
      game_play(game, agent1, agent2) # Play game
      save_to_json(game.outcomes, f"{tag}_gameplay_outcomes")
      print(f"Finished Evaluation of {tag}")

      del agent1, agent2 


def evaluate_against_fixed_opponent(m_paths, full_config, saving_tags, seed = 1):
    
    # Instantiate configs
    game_params = GameParams(**full_config["game_parameters"])
    obs_manager_config = ObservationManagerConfig(**full_config["obs_manager_parameters"])
    ppo_agent_config = EvalAgentConfig(**full_config["ppo_agent_parameters"])
    fixed_agent_config = FixedAgentConfig(**full_config["fixed_agent_config"])
    print("Ready to start experiments")

    set_seed(seed)

    for m, tag in zip(m_paths, saving_tags): 
      # Overwrite adapter configs and initialise agents
      ppo_agent_config.adapter_path = m
      agent1 = EvaluationAgent(ppo_agent_config)
      agent2 = FixedAgent(fixed_agent_config)
      print("Agents initialised.")

      # Initialise game
      game = IteratedMatrixGame(game_params, obs_manager_config, obs_manager_config)
      game_play(game, agent1, agent2, fixed_opponent=True) # Play game
      save_to_json(game.outcomes, f"{tag}_gameplay_outcomes")
      print(f"Finished Evaluation of {tag}")

      del agent1, agent2 


def extract_kl_divergences(master_path, tags: List[str], target="('C', 'D')"):
    dfs = []

    def kl_divergence(p, q):
      p = np.array(p, dtype=np.float64)
      q = np.array(q, dtype=np.float64)

      # Avoid division by zero or log(0)
      p = np.clip(p, 1e-10, None)
      q = np.clip(q, 1e-10, None)

      # Normalize
      p /= p.sum()
      q /= q.sum()

      return np.sum(p * np.log(p / q))

    for ind, tag in enumerate(tags): 
        # Assuming df is your dataframe with columns 'prompt', 'a1', 'a2'
        df = pd.read_csv(master_path(tag))
        if isinstance(target, str): 
          reference_distribution = df[df['prompt'] == target].values[0,1:]# You'll need to replace this
        elif isinstance(target, list):
          reference_distribution = [target[0][ind], target[1][ind]]
        else:
          raise TypeError("target must be either a string or a list.")
          
        df['kl_divergence'] = df.apply(lambda row: kl_divergence([row['a1'], row['a2']], reference_distribution), axis=1)
        dfs.append(df)

    kl_columns = [df['kl_divergence'] for df in dfs]
    kl_array = np.stack(kl_columns, axis=1)
    avg_kl = kl_array.mean(axis=1)

    result_df = pd.DataFrame({'prompt': dfs[0]['prompt'],  'average_kl': avg_kl})
    result_df = result_df.sort_values(by='average_kl')

    return result_df


def print_reward_table(mean_all, std_all, mean_illegal, std_illegal, illegal_actions, illegal_actions_std, E=1):
    headers = ["", "All", "Ignoring illegal", "Illegal Actions"]
    row_labels = ["Total", "Episode 1", "Episode 2", "Episode 3"]

    # Format the data cells: "{mean}±{std}" with 2 significant figures
    def format_cell(mean, std):
        return f"{mean:.2f}±{std:.2f}"

    # Build rows
    rows = []
    for i in range(E):
        row = [
            row_labels[i],
            format_cell(mean_all[i], std_all[i]),
            format_cell(mean_illegal[i], std_illegal[i]), 
            format_cell(illegal_actions[i-1], illegal_actions_std[i-1]) 
        ]
        rows.append(row)

    # Determine column widths
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(4)]

    # Print the table
    def print_row(row):
        print(" | ".join(f"{cell:^{col_widths[i]}}" for i, cell in enumerate(row)))

    print_row(headers)
    print("-" * (sum(col_widths) + 6))
    for row in rows:
        print_row(row)


def extract_evaluation_rewards(path, T:int, E:int, n_games:int, model_number:int=1, game_name:str="PD", return_stats:bool=True):
    if game_name == "PD": 
        mapping = {0: 3, 1: 0, 2: 4, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: -1}
        mapping_ignore_illegal = {0: 3, 1: 0, 2: 4, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: np.nan}
    elif game_name == "MP": 
        mapping = {0: 1, 1: -1, 2: -1, 3: 1, 4: -2, 5: np.nan} if model_number == 1 else {0: -1, 1: 1, 2: 1, 3: -1, 4: np.nan, 5: -2}
        mapping_ignore_illegal = {0: 1, 1: -1, 2: -1, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: -1, 1: 1, 2: 1, 3: -1, 4: np.nan, 5: np.nan}
    elif game_name == "Chicken": 
        mapping = {0: 2, 1: 1, 2: 3, 3: 0, 4: -1, 5: np.nan} if model_number == 1 else {0: 2, 1: 3, 2: 1, 3: 0, 4: np.nan, 5: -1}
        mapping_ignore_illegal = {0: 2, 1: 1, 2: 3, 3: 0, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 2, 1: 3, 2: 1, 3: 0, 4: np.nan, 5: np.nan}
    elif game_name == "Stag": 
        mapping = {0: 4, 1: 0, 2: 3, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 4, 1: 0, 2: 3, 3: 1, 4: np.nan, 5: -1}
        mapping_ignore_illegal = {0: 4, 1: 0, 2: 3, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 4, 1: 0, 2: 3, 3: 1, 4: np.nan, 5: np.nan}
    else: 
        raise ValueError(f"Expected game_name in ['PD', 'MP', 'Chicken', 'Stag'] got {game_name} instead.")

    outcomes = load_from_json(path)
    rewards = np.array([mapping.get(x, x) for x in outcomes]).reshape((-1, T*n_games))
    rewards_no_ill = np.array([mapping_ignore_illegal.get(x, x) for x in outcomes]).reshape((-1, T*n_games))

    mean_r, std_r, mean_no_ill_r, std_no_ill_r = np.zeros(E+1), np.zeros(E+1), np.zeros(E+1), np.zeros(E+1)
    illegal_actions = np.mean(np.array(outcomes).reshape(-1, T*n_games) == 4., axis=1) if model_number == 1 else np.mean(np.array(outcomes).reshape(-1, T*n_games) == 5., axis=1)
    illegal_actions_std = np.std(np.array(outcomes).reshape(-1, T*n_games) == 4., axis=1) if model_number == 1 else np.std(np.array(outcomes).reshape(-1, T*n_games) == 5., axis=1)

    mean_r[0], std_r[0] = np.nanmean(rewards), np.nanstd(rewards)
    mean_no_ill_r[0], std_no_ill_r[0] = np.nanmean(rewards_no_ill), np.nanstd(rewards_no_ill)

    mean_r[1:], std_r[1:] = np.nanmean(rewards, axis=1), np.nanstd(rewards, axis=1)
    mean_no_ill_r[1:], std_no_ill_r[1:] = np.nanmean(rewards_no_ill, axis=1), np.nanstd(rewards_no_ill, axis=1)

    if return_stats:
        return mean_r, std_r, mean_no_ill_r, std_no_ill_r, illegal_actions, illegal_actions_std
    else: 
        print_reward_table(mean_r, std_r, mean_no_ill_r, std_no_ill_r, illegal_actions, illegal_actions_std)



def evaluation_over_seeds(path, model_ids, T, E, n_games, model_number, game_name): 
    means_per_seed, means_illegal_per_seed, illegal_actions_per_seed = np.zeros((len(model_ids), 4)), np.zeros((len(model_ids), 4)), np.zeros((len(model_ids), 3))

    for ind, i in enumerate(model_ids): 
        mean_all, _, mean_illegal, __, illegal_actions, ___ = extract_evaluation_rewards(path(i), T, E, n_games, model_number, game_name)
        means_per_seed[ind, :] = mean_all
        means_illegal_per_seed[ind, :] = mean_illegal
        illegal_actions_per_seed[ind, :] = illegal_actions
    
    means, std, means_illegal, std_illegal = means_per_seed.mean(axis=0), means_per_seed.std(axis=0), means_illegal_per_seed.mean(axis=0), means_illegal_per_seed.std(axis=0)
    illegal_actions, illegal_actions_std = illegal_actions_per_seed.mean(axis=0), illegal_actions_per_seed.std(axis=0)
    print_reward_table(means, std, means_illegal, std_illegal, illegal_actions, illegal_actions_std, E=E+1)


def extract_avg_reward_over_seeds(path, T:int, E:int, n_games:int, model_ids: List, model_number:int=1, game_name:str="PD"):
    if game_name == "PD": 
        mapping = {0: 3, 1: 0, 2: 4, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: -1}
        mapping_ignore_illegal = {0: 3, 1: 0, 2: 4, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: np.nan}
    elif game_name == "MP": 
        mapping = {0: 1, 1: -1, 2: -1, 3: 1, 4: -2, 5: np.nan} if model_number == 1 else {0: -1, 1: 1, 2: 1, 3: -1, 4: np.nan, 5: -2}
        mapping_ignore_illegal = {0: 1, 1: -1, 2: -1, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: -1, 1: 1, 2: 1, 3: -1, 4: np.nan, 5: np.nan}
    elif game_name == "Chicken": 
        mapping = {0: 2, 1: 1, 2: 3, 3: -5, 4: -6, 5: np.nan} if model_number == 1 else {0: 2, 1: 3, 2: 1, 3: -5, 4: np.nan, 5: -6}
        mapping_ignore_illegal = {0: 2, 1: 1, 2: 3, 3: -5, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 2, 1: 3, 2: 1, 3: -5, 4: np.nan, 5: np.nan}
    elif game_name == "Stag": 
        mapping = {0: 4, 1: 0, 2: 3, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 4, 1: 0, 2: 3, 3: 1, 4: np.nan, 5: -1}
        mapping_ignore_illegal = {0: 4, 1: 0, 2: 3, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 4, 1: 0, 2: 3, 3: 1, 4: np.nan, 5: np.nan}
    elif game_name == "PD_coop": 
        mapping = {0: 6, 1: 0, 2: 4, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: -1}
        mapping_ignore_illegal = {0: 6, 1: 0, 2: 4, 3: 1, 4: np.nan, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: np.nan}
    else: 
        raise ValueError(f"Expected game_name in ['PD', 'MP', 'Chicken', 'Stag', 'PD_coop'] got {game_name} instead.")

    mean_rewards = []
    mean_rewards_ill = []
    illegal_actions = []
    for m_id in model_ids: 

      outcomes = load_from_json(path(m_id))

      rewards = np.array([mapping.get(x, x) for x in outcomes])
      mean_rewards.append(np.nanmean(rewards))
      rewards_no_ill = np.array([mapping_ignore_illegal.get(x, x) for x in outcomes])
      mean_rewards_ill.append(np.nanmean(rewards_no_ill))
      illegal = np.mean(np.array(outcomes) == 4.) if model_number == 1 else np.mean(np.array(outcomes) == 5.)
      illegal_actions.append(illegal)

    t_crit = stats.t.ppf(0.95, df=len(model_ids)-1)

    mu_r, std_r = np.nanmean(mean_rewards), t_crit * (np.nanstd(mean_rewards) / np.sqrt(len(model_ids))) 
    mu_r_no_ill, std_r_no_ill = np.nanmean(mean_rewards_ill), t_crit * (np.nanstd(mean_rewards_ill) / np.sqrt(len(model_ids))) 
    mu_ill, std_ill = np.mean(illegal_actions), t_crit * (np.std(illegal_actions) / np.sqrt(len(model_ids)))

    return mu_r, std_r, mu_r_no_ill, std_r_no_ill, mu_ill, std_ill



def evaluation_multiple_models(paths, model_ids, model_numbers, tags, T, E, n_games, game_name:str="PD", return_table=True): 
  assert len(paths) == len(model_ids) == len(model_numbers) == len(tags)
  N = len(paths)

  if type(T) == int: 
    T = [T] * N
  else: 
    assert len(T) == N

  all_mu_r, all_std_r = np.zeros(N), np.zeros(N)
  all_mu_r_no_ill, all_std_r_no_ill = np.zeros(N), np.zeros(N)
  all_mu_ill, all_std_ill = np.zeros(N), np.zeros(N)

  for ind, (path, m_id, model_num, t) in enumerate(zip(paths, model_ids, model_numbers, T)): 
    mu_r, std_r, mu_r_no_ill, std_r_no_ill, mu_ill, std_ill = extract_avg_reward_over_seeds(path, t, E, n_games, m_id, model_num, game_name)
    all_mu_r[ind], all_std_r[ind] = mu_r, std_r
    all_mu_r_no_ill[ind], all_std_r_no_ill[ind] = mu_r_no_ill, std_r_no_ill
    all_mu_ill[ind], all_std_ill[ind] = mu_ill, std_ill
  
  if return_table:
    print_evaluation_multiple_models(all_mu_r, all_std_r, all_mu_r_no_ill, all_std_r_no_ill, all_mu_ill, all_std_ill, tags)
  else: 
    return  (all_mu_r, all_std_r, all_mu_r_no_ill, all_std_r_no_ill, all_mu_ill, all_std_ill)


def print_evaluation_multiple_models(all_mu_r, all_std_r, all_mu_r_no_ill, all_std_r_no_ill, all_mu_ill, all_std_ill, tags):
    headers = ["", "Reward per Step", "Reward per Step (ignoring illegal)", "Percentage of Illegal Actions"]

    # Format the data cells: "{mean}±{std}" with 2 decimal places
    def format_cell(mean, std):
        return f"{mean:.2f}±{std:.2f}"

    # Build rows for each checkpoint
    rows = []
    for i, tag in enumerate(tags):
        row = [
            f"{tag}",
            format_cell(all_mu_r[i], all_std_r[i]),
            format_cell(all_mu_r_no_ill[i], all_std_r_no_ill[i]), 
            format_cell(all_mu_ill[i], all_std_ill[i]) 
        ]
        rows.append(row)

    # Determine column widths
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]

    # Print helper
    def print_row(row):
        print(" | ".join(f"{cell:^{col_widths[i]}}" for i, cell in enumerate(row)))

    # Print the table
    print_row(headers)
    print("-" * (sum(col_widths) + 3 * (len(headers) - 1)))  # separators account for " | "
    for row in rows:
        print_row(row)