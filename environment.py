import numpy as np
import torch

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from agents import PPOAgent, FixedAgent
from observation_managers import ObservationManager


@dataclass
class EnvState:
    """Holds the timestep within the episode (inner_t) and the episode number within the trial (outer_t)"""
    inner_t: int
    outer_t: int


@dataclass
class GameParams():
    """Holds the necessary parameters to initialise the game."""
    t_max: int
    e_max: int
    n_games: int
    r_matrix: List[List[List[float]]]
    penalty: float

    def __post_init__(self): 
        r_matrix = np.array(self.r_matrix)
        assert r_matrix.shape == (2, 2, 2), f"The reward matrix has an incorrect size. Expected (2,2,2), got {r_matrix.shape}"


@dataclass
class StepResults:
    """Holds the output of a single environment step, including rewards, new observations, and new environment state"""
    r1: List[torch.Tensor]
    r2: List[torch.Tensor]
    new_obs1: List[str]
    new_obs2: List[str]
    new_env_state: EnvState


@dataclass
class TrajectoryData: 
    """Holds trajectory data for a given game (full episodes for non-shapers, and full trials for shapers)"""
    query_tensors: List[List[torch.Tensor]] = field(default_factory=list) # The nested list structure is required for PPO Training with trl v.0.11
    response_tensors: List[List[torch.Tensor]] = field(default_factory=list)
    env_ids: List[List[int]] = field(default_factory=list) # Required for GAE
    rewards: List[List[torch.Tensor]] = field(default_factory=list)
    last_observation: List[str] = field(default_factory=list)

    def __post_init__(self):
        assert len(self.query_tensors) == len(self.response_tensors) == len(self.env_ids) == len(self.rewards), "Lengths of query_tensors, response_tensors, env_ids, and rewards must match."

    def _update(self, new_queries: List[torch.Tensor], new_responses: List[torch.Tensor], new_ids: List, new_rewards: List[torch.Tensor]) -> None:
        """Updates the trajectory with results from the current episode. Forbidden transitions are excluded"""
        allowed_transitions = [not torch.isnan(x) for x in new_rewards] # exclude rounds where player took legal action, and opponent took illegal one

        self.query_tensors.append([x for x, cond in zip(new_queries, allowed_transitions) if cond])
        self.response_tensors.append([x for x, cond in zip(new_responses, allowed_transitions) if cond])
        self.env_ids.append([x for x, cond in zip(new_ids, allowed_transitions) if cond])
        self.rewards.append([x for x, cond  in zip(new_rewards, allowed_transitions) if cond])

        assert(len(self.query_tensors) == len(self.response_tensors) == len(self.env_ids) == len(self.rewards))

    def _reset(self, init_obs:str, n_parallel:int) -> None: 
        """Resets queries, responses, rewards, and environment ids, and sets the last observation to be that received at the beginning of the game"""
        self.query_tensors, self.response_tensors, self.env_ids, self.rewards = [], [], [], []
        self.last_observation = [init_obs] * n_parallel


class TokenToActionMapper: 
    """This mapper uses the same legal tokens for all parallel games"""
    def __init__(self, agent_id:str, a1_tok:int=7, a2_tok:int=8):
        self.agent_id = agent_id
        self.a1_tok, self.a2_tok = a1_tok, a2_tok 

    def map(self, responses: List[torch.Tensor]) -> torch.Tensor:
        """Maps generated tokens to actions, where 0 := action 1, 1 := action 2, 2 := any other illegal action"""
        response_tensor = torch.tensor(responses)

        actions = torch.full(response_tensor.shape, fill_value=2) # Illegal actions default to 2
        actions[response_tensor == torch.full(response_tensor.shape, self.a1_tok)] = 0
        actions[response_tensor == torch.full(response_tensor.shape, self.a2_tok)] = 1

        return actions


class IteratedMatrixGame:

    def __init__(self, game_params:GameParams, *obs_manager_params): 
        """Two-player, two-action matrix game simulation"""
        # Check number of observation manager configs does not exceed maximum number of players.
        assert (len(obs_manager_params) <= 2), f"Number of observation manager configurations exceeds maximum number of players. Expected 2, got {len(obs_manager_params)}."

        self.t_max, self.e_max, self.n_games = game_params.t_max, game_params.e_max, game_params.n_games

        # Initialise observation managers and token to action maps for each player
        self.obs_managers, self.token_action_maps = {}, {}
        for ind, p in enumerate(obs_manager_params):
            self.obs_managers[f"agent_{ind+1}"] = ObservationManager(p) 
            self.token_action_maps[f"agent_{ind+1}"] = TokenToActionMapper(agent_id = f"{ind}", a1_tok=p.a1_tok, a2_tok=p.a2_tok) 

        # Convert the input reward matrix into an outcome reward matrix
        self.r_matrix = self._reshape_reward_matrix(game_params.r_matrix, game_params.penalty)
        self.action_pair_lookup = torch.tensor([[0, 1, 5], [2, 3, 5], [4, 4, 4]]) # Maps action pairs to outcomes
        self.outcomes = [] # Initalise containers to track outcomes of the games


    def _reshape_reward_matrix(self, r_matrix:List[List[float]], penalty:float) -> torch.Tensor:
        """Converts game reward matrix into an outcome matrix that accounts for illegal actions from both players."""
        outcome_matrix = torch.full((2, 6), float('nan')) # Initialise outcome matrix. If opponent plays illegal action, and player plays legal, the result is nan. 
        outcome_matrix[:, :4] = torch.tensor(r_matrix).view(2, -1) 
        outcome_matrix[:, 4] = penalty  

        return outcome_matrix

    def step(self, obs1:List[str], obs2:List[str], response_tensor1: List[torch.Tensor], response_tensor2: List[torch.Tensor], env_state: EnvState, agent1_learner:bool=True, agent2_learner:bool=True) -> StepResults:
        """Returns new environment state and observations for both players given a set of observations and actions"""
        assert len(response_tensor1) == len(obs1) == len(response_tensor2), "The number of actions and observations should be the same for all players"
        
        if agent2_learner: 
            assert len(response_tensor1) == len(obs2)

        t, e = env_state.inner_t, env_state.outer_t
        t += 1

        # Map token IDs (response_tensors) to actions (0,1,2)
        a1 = self.token_action_maps["agent_1"].map(response_tensor1)
        a2 = self.token_action_maps["agent_2"].map(response_tensor2)

        # Obtain outcomes - List[tensor]. Contains two tensors of shape (num_responses, )
        outcomes = [self.action_pair_lookup[a1, a2], self.action_pair_lookup[a2, a1]]
        self.outcomes.extend(outcomes[0].tolist())

        # Get rewards
        r1, r2 = self.r_matrix[0, outcomes[0]].unsqueeze(1), self.r_matrix[1, outcomes[1]].unsqueeze(1)# Rewards - torch.tensor of shape (num_responses, )

        # Update internal state
        if t == self.t_max:
            t, e = 0, e+1

        if e == self.e_max:
            t, e = 0, 0
    
        new_env_state = EnvState(inner_t=t, outer_t=e)

        # Get new observations
        new_obs1 = self.obs_managers["agent_1"].batch_update_obs(obs1, [a1, a2], new_env_state.inner_t, new_env_state.outer_t) if agent1_learner else []
        new_obs2 = self.obs_managers["agent_2"].batch_update_obs(obs2, [a2, a1],  new_env_state.inner_t, new_env_state.outer_t) if agent2_learner else []
        
        return StepResults(r1 = list(r1), r2 = list(r2), new_obs1 = new_obs1, new_obs2 = new_obs2, new_env_state = new_env_state)


def inner_rollout_fixed_opponent(game:IteratedMatrixGame, env_state: EnvState, traj_data1: TrajectoryData, agent1: PPOAgent, agent2: FixedAgent) -> Tuple[TrajectoryData, EnvState]:
    """Simulates one full episode of length t_max between an LLM based agent and a fixed opponent"""
    assert len(traj_data1.last_observation) == game.n_games == agent2.n_games # Check both agents are playing the same number of parallel games
    assert(env_state.inner_t == 0) # Check the episode is starting

    obs1 = traj_data1.last_observation # Retrieve last observation 
    agent1_env_ids = list(range(game.n_games)) # Initialise environment ids

    for no_interactions in range(game.t_max):

        print(f"\nInteraction {no_interactions+1}.")

        # Generate Player 1's actions
        query_tensor1 = agent1.tokenize_observation(obs1)
        response_tensors1 = agent1.take_action(query_tensor1) 

        # Generate Player 2's actions
        generation_params = {}
        if agent2.agent_type == "TFT":
            actions = [(game.token_action_maps["agent_1"].map(response)).tolist() for response in traj_data1.response_tensors]
            generation_params["opp_actions"] = torch.tensor(actions)
        response_tensors2 = agent2.take_action(generation_params) 

        # Get rewards
        next_step = game.step(obs1, [], response_tensors1, response_tensors2, env_state, True, False)
        obs1, env_state = next_step.new_obs1, next_step.new_env_state

        # Update trajectory
        traj_data1._update(new_queries=query_tensor1, new_responses=response_tensors1, new_ids=agent1_env_ids, new_rewards=next_step.r1)

    traj_data1.last_observation = obs1 # Update last obseration 

    # Print the current counts
    episode_outcomes = torch.tensor(game.outcomes[-game.t_max*game.n_games:]) 
    print(f"In this episode - CC: {(episode_outcomes == 0).sum().item()}, CD: {(episode_outcomes == 1).sum().item()}, DC: {(episode_outcomes == 2).sum().item()}, DD: {(episode_outcomes == 3).sum().item()}.\n I: {(episode_outcomes == 4).sum().item()}")

    assert(env_state.inner_t == 0) # Check a full episode has been completed
    return (traj_data1, env_state)


def inner_rollout(game: IteratedMatrixGame, env_state: EnvState, traj_data1: TrajectoryData, traj_data2: TrajectoryData, agent1: PPOAgent, agent2: PPOAgent) -> Tuple[TrajectoryData, TrajectoryData, EnvState]: 
    """Rollout trajectories of two LLM-based agents playing a specified Iterated Matrix Game"""

    obs1, obs2 = traj_data1.last_observation, traj_data2.last_observation # Retrieve last observations for both players
    assert len(obs1) == len(obs2) == game.n_games # Check both agents are playing the same number of parallel games
    assert(env_state.inner_t == 0) # Check the environment state is consistent 

    # Initialise environment ids - currently, for shaper, do advantage estimation with FULL TRIAL TRAJECTORY
    agent1_env_ids = list(range(game.n_games)) # list(range(e*n_games, (e+1) * n_games)) if agent1.is_shaper else list(range(n_games)) for GAE with episode only results
    agent2_env_ids = list(range(game.n_games))

    for no_interactions in range(game.t_max):

        print(f"\nInteraction {no_interactions+1}.")

        
        query_tensor1, query_tensor2 = agent1.tokenize_observation(obs1), agent2.tokenize_observation(obs2) # Tokenized textual observations
        reponse_tensors1, reponse_tensors2 = agent1.take_action(query_tensor1), agent2.take_action(query_tensor2) # Take actions

        next_step = game.step(obs1, obs2, reponse_tensors1, reponse_tensors2, env_state) # Get rewards and observations
        obs1, obs2, env_state = next_step.new_obs1, next_step.new_obs2, next_step.new_env_state

        traj_data1._update(new_queries=query_tensor1, new_responses=reponse_tensors1, new_ids=agent1_env_ids, new_rewards=next_step.r1) # Update player 1's trajectory
        traj_data2._update(new_queries=query_tensor2, new_responses=reponse_tensors2, new_ids=agent2_env_ids, new_rewards=next_step.r2) # Update player 2's trajectory
        
    traj_data1.last_observation, traj_data2.last_observation = obs1, obs2 # Update last obseration 
   
    # Print the current counts
    episode_outcomes = torch.tensor(game.outcomes[-game.t_max*game.n_games:]) 
    print(f"In this episode - CC: {(episode_outcomes == 0).sum().item()}, CD: {(episode_outcomes == 1).sum().item()}, DC: {(episode_outcomes == 2).sum().item()}, DD: {(episode_outcomes == 3).sum().item()}.\n I: {(episode_outcomes == 4).sum().item()}")
    
    assert(env_state.inner_t == 0) # Check a full episode has been completed
    return (traj_data1, traj_data2, env_state)


def outer_rollout(game:IteratedMatrixGame, agent1:PPOAgent, agent2:PPOAgent) -> Tuple[TrajectoryData, TrajectoryData, EnvState]:
    """Rollout a whole trial. If the agents are non-shapers, update the parameters at the end of each episode."""
    
    # Initialise environment state and trajectories
    env_state = EnvState(inner_t=0, outer_t=0) 
    traj_data1 = TrajectoryData(last_observation = [game.obs_managers["agent_1"].game_description + game.obs_managers["agent_1"].instruction_prompt] * game.n_games)
    traj_data2 = TrajectoryData(last_observation = [game.obs_managers["agent_2"].game_description + game.obs_managers["agent_2"].instruction_prompt] * game.n_games)

    for e in range(game.e_max): 

        traj_data1, traj_data2, env_state = inner_rollout(game, env_state, traj_data1, traj_data2, agent1, agent2) # rollout trajectory

        for tag, agent, data in zip(["agent_1", "agent_2"], [agent1, agent2], [traj_data1, traj_data2]):
          if not agent.is_shaper:
              agent.update_parameters(data) # update agent parameters
              agent.update_vf_coef() # Update VF coefficient
              data._reset(game.obs_managers[tag].game_description + game.obs_managers[tag].instruction_prompt , game.n_games) # reset trajectories after update is completed
    
    assert (env_state.inner_t == 0) and (env_state.outer_t == 0) # Check the trial has ended successfully

    return (traj_data1, traj_data2, game.outcomes)
