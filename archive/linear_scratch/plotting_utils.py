import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy.stats as stats

from typing import List, Dict, Callable
from utils.file_management_utils import load_from_json


def merge_dicts(dict_list:Dict): 

   merged_dict = {key: np.array([dic[key] for dic in dict_list]) for key in list(dict_list[0].keys())}
   return merged_dict


def plot_individual_training_stat(file_path, model_IDs, saving_path, stat_name, stat_label, avg = False, avg_params = {"avg_label": None, "avg_color": "dodgerblue"}, save = False, colors = ["dodgerblue", "darkblue", "lightblue", "indianred", "coral"]):

   data = [load_from_json(file_path(ID)) for ID in model_IDs] # Load the data
   no_exp, no_steps = len(data), list(range(1, len(data[0]["total_loss"])+ 1))
   
   if avg: 
      merged_data = merge_dicts(data) # Merge into a single dict: shape no_exp x no_steps
      mean, std = merged_data[stat_name].mean(axis=0), merged_data[stat_name].std(axis=0)
      t_crit = stats.t.ppf(0.975, df=no_exp-1)
      plt.plot(no_steps, mean, color = avg_params["avg_color"], label = avg_params["avg_label"])
      plt.fill_between(no_steps, mean - t_crit * (std / np.sqrt(no_exp)), mean + t_crit * (std / np.sqrt(no_exp)), color = avg_params["avg_color"], alpha = 0.4)
      saving_extension = "avg_"
   else: 
      for ind, d in enumerate(data):
         plt.plot(no_steps, d[stat_name], color = colors[ind], label = f"Seed = {ind+1}")
      saving_extension = "ind_"

   plt.xlabel("Number of steps")
   plt.ylabel(stat_label)
   plt.legend()

   if save:
      plt.savefig(saving_path + f"{saving_extension}{stat_label}.pdf", dpi=300)


def get_loss_ratio(file_path, models_ids, color = "dodgerblue", label = None): 
    data = [load_from_json(file_path(ID)) for ID in models_ids] # Load the data
    no_exp, no_steps = len(data), list(range(1, len(data[0]["total_loss"])+ 1))
    ratios = np.zeros((no_exp, len(data[0]["total_loss"])))
    for ind, exp_data in enumerate(data): 
        policy_loss, value_loss = exp_data["policy_loss"], exp_data["value_loss"]
        ratios[ind, :] = abs(np.array(policy_loss)/ np.array(value_loss))
    mean, std = ratios.mean(axis=0), ratios.std(axis=0)
    t_crit = stats.t.ppf(0.975, df=no_exp-1)
    plt.plot(no_steps, mean, color = color, label = label)
    plt.fill_between(no_steps, np.maximum(0, mean - t_crit * (std / np.sqrt(no_exp))), mean + t_crit * (std / np.sqrt(no_exp)), color = color, alpha = 0.4)


def get_adaptive_c_vf(file_path, models_ids, alpha, target, initial_c_vf=0.1, color = "dodgerblue", label = None): 
    data = [load_from_json(file_path(ID)) for ID in models_ids] # Load the data
    no_exp, no_steps = len(data), list(range(1, len(data[0]["total_loss"])+ 1))
    coefficients = np.zeros((no_exp, len(data[0]["total_loss"])))
    coefficients[:, 0] = 0.1

    for ind, exp_data in enumerate(data): 
        policy_loss, value_loss = exp_data["policy_loss"], exp_data["value_loss"]
        # Now calculate coefficients
        for ind_2, (l_p, l_vf) in enumerate(zip(policy_loss[:-1], value_loss[:-1])):
            coefficients[ind, ind_2+1] = max(coefficients[ind, ind_2] + alpha * (l_p / (target*l_vf) - coefficients[ind, ind_2]), 0)

    mean, std = coefficients.mean(axis=0), coefficients.std(axis=0)
    t_crit = stats.t.ppf(0.975, df=no_exp-1)
    plt.plot(no_steps, mean, color = color, label = label)
    plt.fill_between(no_steps, np.maximum(0, mean - t_crit * (std / np.sqrt(no_exp))), mean + t_crit * (std / np.sqrt(no_exp)), color = color, alpha = 0.4)


def compare_adaptive_c_vf(file_paths, model_ids, alphas, targets, initial_c_vfs, colors=["dodgerblue", "indianred"], labels=["Default", "Adaptive"]): 
    for file_path, color, label, alpha, target, initial_c_vf in zip(file_paths, colors, labels, alphas, targets, initial_c_vfs): 
        get_adaptive_c_vf(file_path, model_ids, alpha, target, initial_c_vf, color, label)
    plt.legend()


def compare_loss_ratios(file_paths, model_ids, colors=["dodgerblue", "indianred"], labels=["Default", "Adaptive"]): 
    for file_path, color, label in zip(file_paths, colors, labels): 
        get_loss_ratio(file_path, model_ids, color, label)
    plt.legend()


def compare_avg_statistic(file_paths, model_IDs, saving_path, stat_name, stat_label, avg_labels, save = False, colors = ["dodgerblue", "darkblue", "coral", "indianred", "rebeccapurple", "brown", "magenta"]):

   for file_path, label, color in zip(file_paths, avg_labels, colors): 
      avg_params = {"avg_label": label, "avg_color": color}
      plot_individual_training_stat(file_path, model_IDs, saving_path, stat_name, stat_label, avg = True, avg_params = avg_params, save = False)
   
   if save:
      plt.savefig(saving_path + f"{stat_label}_comparison.jpeg", dpi=300)


def extract_outcomes_per_iteration(outcomes:List[int], t:int, n:int):
    """Extract percentage of outcomes per epoch"""
    slices = [slice(start, start + t*n) for start in range(0, len(outcomes), t*n)]
    ccs, cds, dcs, dds, illegal = [], [], [], [], []

    for sl in slices:
        round_outcomes = np.array(outcomes[sl])

        ccs.append((round_outcomes == 0).sum() / (n*t))
        cds.append((round_outcomes == 1).sum() / (n*t))
        dcs.append((round_outcomes == 2).sum() / (n*t))
        dds.append((round_outcomes == 3).sum() / (n*t))
        illegal.append(((round_outcomes == 5).sum() + (round_outcomes == 4).sum() )/ (n*t)) #
    
    return ccs, cds, dcs, dds, illegal


def extract_rewards_from_outcomes(path, T:int, E:int, n_games:int, model_number:int=1, game_name:str="PD"): 
    if game_name == "PD": 
        mapping = {0: 3, 1: 0, 2: 4, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: -1}
    elif game_name == "C-IPD": 
        mapping = {0: 6, 1: 0, 2: 4, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 3, 1: 4, 2: 0, 3: 1, 4: np.nan, 5: -1}
    elif game_name == "MP": 
        mapping = {0: 1, 1: -1, 2: -1, 3: 1, 4: -2, 5: np.nan} if model_number == 1 else {0: -1, 1: 1, 2: 1, 3: -1, 4: np.nan, 5: -2}
    elif game_name == "MP_switched": 
        mapping = {0: -1, 1: 1, 2: 1, 3: -1, 4: -2, 5: np.nan} if model_number == 1 else {0: 1, 1: -1, 2: -1, 3: 1, 4: np.nan, 5: -2}
    elif game_name == "Chicken": 
        mapping = {0: 2, 1: 1, 2: 3, 3: -5, 4: -6, 5: np.nan} if model_number == 1 else {0: 2, 1: 3, 2: 1, 3: -5, 4: np.nan, 5: -6}
    elif game_name == "Stag": 
        mapping = {0: 4, 1: 0, 2: 3, 3: 1, 4: -1, 5: np.nan} if model_number == 1 else {0: 4, 1: 3, 2: 0, 3: 1, 4: np.nan, 5: -1}
    else: 
        raise ValueError(f"Expected game_name in ['PD', 'MP', 'Chicken] got {game_name} instead.")
    outcomes = load_from_json(path)
    mean_rewards = []

    for i in range(0, len(outcomes), T*E*n_games): 
        epoch_rewards = [mapping.get(x, x) for x in outcomes[i:i+T*E*n_games]]
        mean_rewards.append(np.nanmean(epoch_rewards))
    return mean_rewards


def plot_unscaled_rewards(master_path, model_ids:List[int], t:int, e:int, n_games:int, no_epochs:int, model_number:int=1, game_name:str="PD", color:str="dodgerblue", axis=None, label:str=None):
    """Plot the average per step reward per epoch over seeds for the same run"""

    if game_name == "PD":
        maximum_val, minimum_val = 4.0, -1.0
    elif game_name == "C-IPD": 
        maximum_val, minimum_val = 6.0, -2.0
    elif game_name == "MP" or game_name == "MP_switched": 
        maximum_val, minimum_val = 1.0, -2.0
    elif game_name == "Chicken": 
        maximum_val, minimum_val = 3.0, -6.0
    elif game_name == "Stag": 
        maximum_val, minimum_val = 4.0, -1.0

    n = len(model_ids)
    rewards = np.array([extract_rewards_from_outcomes(master_path(id), t, e, n_games, model_number, game_name) for id in model_ids])
    assert rewards.shape == (n, no_epochs), print(rewards.shape)

    mean, std = np.mean(rewards, axis=0), np.std(rewards, axis=0)
    t_crit = stats.t.ppf(0.95, df=n-1)

    print(mean[-1], std[-1])

    if axis: 
        axis.plot(list(range(no_epochs)), mean, color= color, label=label)
        axis.fill_between(list(range(no_epochs)), np.maximum(minimum_val, mean - t_crit * (std / np.sqrt(len(model_ids)))), np.minimum(mean + t_crit * (std / np.sqrt(len(model_ids))), maximum_val), color= color, alpha = 0.5)
    else: 
        plt.plot(list(range(no_epochs)), mean, color= color, label=label)
        plt.fill_between(list(range(no_epochs)), np.maximum(minimum_val, mean - t_crit * (std / np.sqrt(len(model_ids)))), np.minimum(mean + t_crit * (std / np.sqrt(len(model_ids))), maximum_val), color= color, alpha = 0.5)


def compare_unscaled_rewards(file_paths, model_ids:List[int], labels:List[str], model_numbers: List[int], t:int, e:int, n_games:int, no_epochs:int, game_names=None, saving_path:str=None, axis = None, x_label = False, y_label = False, colors = ["dodgerblue", "darkblue", "coral", "indianred", "rebeccapurple", "brown", "magenta"]):
    
    if game_names is None:
        game_names = ["PD"] * len(file_paths)
    
    if type(n_games) == int: 
        n_games = [n_games] * len(file_paths)
    
    if type(t) == int: 
        t = [t] * len(file_paths)

    for ind, (path, color, label, model_number, game_name, n_game, t_ind) in enumerate(zip(file_paths, colors, labels, model_numbers, game_names, n_games, t)):
        num_e = e if (type(e) is int) else e[ind]
        plot_unscaled_rewards(path, model_ids, t_ind, num_e, n_game, no_epochs, model_number=model_number, game_name=game_name, color=color, label=label, axis=axis)

    if axis: 
        axis.legend()
        if x_label: 
            axis.set_xlabel("Epochs")
        if y_label:
            axis.set_ylabel("Average Reward per step")
    
    else: 
        plt.legend()
        if x_label: 
            plt.xlabel("Epochs")
        if y_label: 
            plt.ylabel("Average Reward per step")
    if saving_path:
        plt.save(saving_path, dpi=300)


def plot_episode_outcomes(*file_paths, t, n_games, no_epochs, colors = ["dodgerblue", "darkblue", "coral", "indianred", "purple"], save=False, axis = None, keys = None, x_label = False, y_label = False):
    """Plot episode outcomes averaged over runs"""
    
    if keys is None: 
        keys = ["CC", "CD", "DC", "DD", "I"]
    data = {key: np.zeros((len(file_paths), no_epochs)) for key in keys}
    no_exp, no_steps = len(file_paths), list(range(1, no_epochs+ 1))

    for ind, path in enumerate(file_paths):

        # Load file
        outcomes = load_from_json(path)
        results = extract_outcomes_per_iteration(outcomes, t, n_games)
        
        for key, val in zip(keys, results):
            data[key][ind, :] = val

    for (key, val), color in zip(data.items(), colors): 
        mean, std = val.mean(axis=0), val.std(axis=0)
        t_crit = stats.t.ppf(0.99, df=no_exp-1)
        print(key, mean[-1])
        if axis: 
            axis.plot(no_steps, mean, color = color, label = key)
            axis.fill_between(no_steps, np.maximum(0, mean - t_crit * (std / np.sqrt(len(file_paths)))), np.minimum(mean + t_crit * (std / np.sqrt(len(file_paths))), 1), color = color, alpha = 0.5)
        else: 
            plt.plot(no_steps, mean, color = color, label = key)
            plt.fill_between(no_steps, np.maximum(0, mean - t_crit * (std / np.sqrt(len(file_paths)))), np.minimum(mean + t_crit * (std / np.sqrt(len(file_paths))), 1), color = color, alpha = 0.5)

    if axis: 
        axis.legend()
        if x_label: 
            axis.set_xlabel("Epoch")
        if y_label: 
            axis.set_ylabel("State Visitation")

    else: 
        plt.legend()
        if x_label: 
            plt.xlabel("Epoch")
        if y_label:
            plt.ylabel("State Visitation")


def plot_illegal_actions(file_paths, model_IDs, all_no_epochs, all_no_games, all_ts, labels, title=None,  saving_path = None,  colors = ["dodgerblue", "darkblue", "coral", "indianred", "rebeccapurple", "brown", "magenta"]): 
    
    for file_path, no_epochs, n_games, t, label, color in zip(file_paths, all_no_epochs, all_no_games, all_ts, labels, colors): 
        illegal_actions =  np.zeros((len(model_IDs), no_epochs))
        for ind, model_id in enumerate(model_IDs):
            outcomes = load_from_json(file_path(model_id))
            ccs, cds, dcs, dds, illegal = extract_outcomes_per_iteration(outcomes, t, n_games)
            illegal_actions[ind, :] = illegal
        mean, std = illegal_actions.mean(axis=0), illegal_actions.std(axis=0)
        t_crit = stats.t.ppf(0.975, df=len(model_IDs)-1)
        plt.plot(list(range(no_epochs)), mean, color = color, label = label)
        plt.fill_between(list(range(no_epochs)), np.maximum(0, mean - t_crit * (std / np.sqrt(len(model_IDs)))), np.minimum(1.0, mean + t_crit * (std / np.sqrt(len(model_IDs)))), color = color, alpha = 0.5)
    plt.legend()
    if title: 
        plt.title(title)
    if saving_path:
        plt.savefig(saving_path + f"{stat_label}_comparison.jpeg", dpi=300)


def plot_cooperation_prob(file_paths, model_IDs, all_no_epochs, all_no_games, all_ts, labels, model_number=[1], title=None,  saving_path = None,  colors = ["dodgerblue", "indianred", "darkblue", "coral",  "rebeccapurple", "brown", "magenta"]): 
    
    for file_path, no_epochs, n_games, t, label, color, m_num in zip(file_paths, all_no_epochs, all_no_games, all_ts, labels, colors, model_number): 
        cooperation_percentage =  np.zeros((len(model_IDs), no_epochs))
        for ind, model_id in enumerate(model_IDs):
            outcomes = load_from_json(file_path(model_id))
            ccs, cds, dcs, dds, illegal = extract_outcomes_per_iteration(outcomes, t, n_games)
            cooperation_percentage[ind, :] = np.array(ccs) + np.array(cds) if m_num == 1 else np.array(ccs) + np.array(dcs)
        mean, std = cooperation_percentage.mean(axis=0), cooperation_percentage.std(axis=0)
        t_crit = stats.t.ppf(0.975, df=len(model_IDs)-1)
        plt.plot(list(range(no_epochs)), mean, color = color, label = label)
        plt.fill_between(list(range(no_epochs)), np.maximum(0, mean - t_crit * (std / np.sqrt(len(model_IDs)))), np.minimum(1.0, mean + t_crit * (std / np.sqrt(len(model_IDs)))), color = color, alpha = 0.5)

    plt.legend()
    if title: 
        plt.title(title)
    if saving_path:
        plt.savefig(saving_path + f"{stat_label}_comparison.jpeg", dpi=300)


def compare_rewards_per_trial(path, model_IDs, colors = ["dodgerblue", "indianred"], title=None): 

    data_m1 = [load_from_json(path(1, ID)) for ID in model_IDs]
    data_m2 = [load_from_json(path(2, ID)) for ID in model_IDs]

    # Get number of experiments, trials, and epochs
    n_exp, n_trials, n_steps = len(data_m1), len(data_m1[0]["mean_scores"]), len(data_m2[0]["mean_scores"])
    e_max = n_steps // n_trials

    # Merge data
    merged_data_m1 = merge_dicts(data_m1) # Merge into a single dict: shape no_exp x no_steps
    mean_m1, std_m1 = merged_data_m1["mean_scores"].mean(axis=0), merged_data_m1["mean_scores"].std(axis=0)

    # Convert rewars per episode into rewards per trial for model 2
    merged_data_m2 = merge_dicts(data_m2)
    m2_per_trial = np.array([np.mean(merged_data_m2["mean_scores"][:, i:i+e_max], axis = 1) for i in range(0, n_steps, e_max)])
    mean_m2, std_m2 = np.mean(m2_per_trial, axis = 1), np.std(m2_per_trial, axis = 1)
    t_crit = stats.t.ppf(0.975, df=len(model_IDs)-1)

    plt.plot(list(range(n_trials)), mean_m1, color = colors[0], label = "Model 1")
    plt.fill_between(list(range(n_trials)), np.maximum(-1.0, mean_m1 - t_crit * (std_m1 / np.sqrt(len(model_IDs)))), np.minimum(4.0, mean_m1 + t_crit * (std_m1 / np.sqrt(len(model_IDs)))), color =  colors[0], alpha = 0.5)

    plt.plot(list(range(n_trials)), mean_m2, color = colors[1], label = "Model 2")
    plt.fill_between(list(range(n_trials)), np.maximum(-1.0, mean_m2 - t_crit * (std_m2 / np.sqrt(len(model_IDs)))), np.minimum(4.0, mean_m2 + t_crit * (std_m2 / np.sqrt(len(model_IDs)))), color =  colors[0], alpha = 0.5)

    plt.legend()
    plt.title(title)
    plt.xlabel("Trial Number")
    plt.ylabel("Mean reward")


def visualize_strategies(data_path:str, saving_path:str=None, img_title:str=None): 
    """Visualize the different strategies obtained via the utils.evaluation_utils.extract_all_legal_token_probabilities function."""

    # Read data anc convert into 2D array for visualisation
    df = pd.read_csv(data_path)
    strategies = df[['a1', 'a2']].values
    #strategy_array = np.array(strategies)
    #print(np.mean(strategy_array, axis=0), np.std(strategy_array, axis=0))

    # Create a histogram
    hist, xedges, yedges = np.histogram2d(strategies[:, 0], strategies[:, 1], bins=20, range=[[0, 1], [0, 1]])
    # Plot the heatmap
    plt.imshow(hist, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.colorbar()
    plt.xlabel('Probability of Defecting')
    plt.ylabel('Probability of Cooperating')
    plt.title(img_title)
    if saving_path:
        plt.savefig(f"{saving_path}.pdf", dpi=300)
    
    else: 
        plt.show()


def plot_evaluation_rewards(master_path:Callable[[str], str], extensions: List[str], saving_path:str=None): 
    """Only to be used with data generated by the utils.evaluation_utils.game_play function."""

    for ext in extensions:
        data = load_from_json(master_path(ext)) # Load data

        mean_rewards1, mean_rewards2 = [], []

        max_rounds1 = max(len(game_rewards) for game_rewards in data["agent_1"])
        max_rounds2 = max(len(game_rewards) for game_rewards in data["agent_2"])

        print(len(data["agent_1"][0]))

        # Iterate through rounds
        for round_id in range(max_rounds1):
            round_rewards = [game_rewards[round_id] if round_id < len(game_rewards) else np.nan for game_rewards in data["agent_1"]]
            mean_reward = np.nanmean(round_rewards)
            mean_rewards1.append(mean_reward)
        
        # Iterate through rounds
        for round_id in range(max_rounds2):
            round_rewards = [game_rewards[round_id]  if round_id < len(game_rewards) else np.nan for game_rewards in data["agent_2"]]
            mean_reward = np.nanmean(round_rewards)
            mean_rewards2.append(mean_reward)

        print(len(mean_rewards2), len(mean_rewards1))

    
    return mean_rewards1, mean_rewards2

