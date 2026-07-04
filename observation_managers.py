import re

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple  

@dataclass
class ObservationManagerConfig:
    """Holds required attributes for the observation manager"""
    a1_tok: int
    a2_tok: int
    a1_string: str
    a2_string: str
    reward_matrix: List[List[List[float]]]
    is_shaper: bool 
    game_description: Optional[str] = "You are playing a 2-player game with actions: {a1_tok}, {a2_tok}. Points are assigned as follows: {a1_tok}/{a1_tok}: {p1_a1a1}/{p2_a1a1},  {a1_tok}/{a2_tok}: {p1_a1a2}/{p2_a2a1}, {a2_tok}/{a1_tok}: {p1_a2a1}/{p2_a1a2}, {a2_tok}/{a2_tok}: {p1_a2a2}/{p2_a2a2}."
    state_prompt: Optional[str] = "In the previous round, you played {own_action} and your opponent played {opp_action}."
    instruction_prompt: Optional[str] = "\nChoose an action for the current round. Reply only with {a1_tok} or {a2_tok}."
    formatting_tags:  Dict = field(default_factory=lambda: {"start_user_tag": "<start_of_turn>user\n", "start_model_tag": "\n<start_of_turn>model\n", "end_tag": "\n<end_of_turn>"})
    additional_info_type: Optional[str] = "state_only"
    model_name: Optional[str] = "gemma-2b"
    transmit_info: Optional[bool] = True

def validate_templates(placeholder_dict, template_list): 
    """Validate that templates contain required placeholders"""
    for template, required, name in zip(template_list, placeholder_dict.values(), placeholder_dict.keys()):
        if template != "table": 
            placeholders = set(re.findall(r"\{(.*?)\}", template))
            if missing := required - placeholders:
                raise ValueError(f"{name} is missing required placeholders: {missing}")

def create_payoff_table(params):
    """Create an appropriate string cotaining a table. params must include the tokens used to express actions 1 and 2, the payoffs for both models."""
    def format_cell(val1, val2, width=7):
                payoff = f"({val1}, {val2})"
                return payoff.center(width)
    header = f'You are playing a 2-player game with actions: {params["a1_tok"]}, {params["a2_tok"]}. Points are assigned as follows:\n\n'
    top_row = f'| {"":6}| {("**"+params["a1_tok"]+"**").center(7)} | {("**"+params["a2_tok"]+"**").center(7)} |\n'
    divider = "|-------|---------|---------|\n"
    row1 = f'| {"**" + params["a1_tok"]+"**":5} | {format_cell(params["p1_a1a1"], params["p2_a1a1"])} | {format_cell(params["p1_a1a2"], params["p2_a2a1"])} |\n'
    row2 = f'| {"**" + params["a2_tok"]+"**":5} | {format_cell(params["p1_a2a1"], params["p2_a1a2"])} | {format_cell(params["p1_a2a2"], params["p2_a2a2"])} |\n'
    return header + top_row + divider + row1 + row2

class AdditionalInfoHandler(ABC):
    """Abstract base class for additional information handling strategies"""
    @abstractmethod
    def update_obs(self, obs: str, a1: str, a2: str, inner_t: int, outer_t:int, state_updater) -> str:
        """Update observation with additional information"""
        pass

    def _remove_state(self, obs:str) -> str:
        """Returns stateless observation"""
        stateless_obs = obs.split(self.state_tag)[0]
        return f"{stateless_obs}{self.instruction}"
    
    def _extract_outcome_from_state(self, obs:str) -> Tuple[List[str], List[str]]:
        """Extract last round's actions from state description."""
        pattern =  self.state_prompt.format(opp_action=r"(\w+)", own_action=r"(\w+)")

        if match := re.search(pattern, obs):
            return [match.group(1)], [match.group(2)]
        else:
            return [],[]


class BasicStateUpdater(AdditionalInfoHandler):  # Inherit from abstract class
    """Handles state updates without additional information tracking."""

    def __init__(self, a1_string:str, a2_string:str):
        self.a1_string, self.a2_string = a1_string, a2_string

    def update_obs(self, obs: str, a1: str, a2: str, inner_t: int, outer_t: int, update_state: Callable[[str, str, str], str]) -> str:
        """Update observation only if actions are legal."""
        if a1 not in {self.a1_string, self.a2_string} or a2 not in {self.a1_string, self.a2_string}:
            return obs  # Invalid actions → No update

        return update_state(obs, a1, a2)  # Update state and return


class StateOccurrenceUpdater(AdditionalInfoHandler):
    """Handles state updates and records state occurrence throughout the game and the trial."""

    def __init__(self, a1_string:str, a2_string:str, state_tag: str, additional_info_tag: str, game_description: str, state_prompt: str, instruction: str):
        self.state_tag, self.additional_info_tag = state_tag, additional_info_tag 

        self.game_description = game_description
        self.state_prompt = state_prompt
        self.instruction = instruction 

        self.a1_string, self.a2_string = a1_string, a2_string
        self.index_map = {f"{self.a1_string}{self.a1_string}": 0, f"{self.a1_string}{self.a2_string}": 1, f"{self.a2_string}{self.a1_string}": 2, f"{self.a2_string}{self.a2_string}": 3}

        self.current_round_prompt = lambda x, y, z, w: f"The occurrence of each state in the current game has been {self.a1_string}{self.a1_string}:{x}, {self.a1_string}{self.a2_string}:{y}, {self.a2_string}{self.a1_string}:{z}, {self.a2_string}{self.a2_string}:{w}."
        self.prev_game_prompt = lambda game, x, y, z, w: f"The occurrence of each state in game {game} was {self.a1_string}{self.a1_string}:{x}, {self.a1_string}{self.a2_string}:{y}, {self.a2_string}{self.a1_string}:{z}, {self.a2_string}{self.a2_string}:{w}."

    def _extract_current_counts(self, obs:str) -> List[int]:
        """Extract state occurrences from observation. If missing, set to 0."""
        return [int(count) for count in re.findall(r":(\d+)", obs)][-4:] if ("current game" in obs) else [0] * 4

    def _get_updated_counts(self, a1s:List[str], a2s:List[str], counts:List[int]) -> List[int]:
        """Returns the updated state counts given an observation and two lists of players actions."""
        for a1, a2 in zip(a1s, a2s):
            counts[self.index_map[f"{a1}{a2}"]] += 1
        return counts

    def _update_obs_counts(self, obs:str, a1:str, a2:str, inner_t:int, outer_t:int) -> str:
        """Returns the updated observation with the provided state counts"""
        current_counts = self._extract_current_counts(obs) # Extract current counts
        a1s, a2s = self._extract_outcome_from_state(obs) # Extract actions from state
        if inner_t == 0: # If a new episode is starting, add the last played actions to state count as well
            a1s.append(a1)
            a2s.append(a2)

        new_counts = self._get_updated_counts(a1s, a2s, current_counts) # Get new counts
        # If starting a new episode, change phrasing to express state count is from a previous game
        replacement = self.current_round_prompt(*new_counts) if inner_t != 0 else self.prev_game_prompt(outer_t, *new_counts)

        if "current game" in obs: # IF current game is in the observation, replace pattern
            pattern = self.current_round_prompt(r"\d+", r"\d+", r"\d+", r"\d+")
            return re.sub(pattern, replacement, obs)

        # If current game is not in observation, add the current game prompt before the state
        before_state, after_state = obs.split(self.state_tag)
        separator = "\n" if (self.additional_info_tag in obs) else self.additional_info_tag
        return f"{before_state}{separator}{replacement}{self.state_tag}{after_state}"

    def update_obs(self, obs: str, a1:str, a2:str, inner_t:int, outer_t:int, update_state: Callable[[str, str, str], str]) -> str: 
        """Updates the state occurrence based on the input state, and updates the input state based on the new actions."""

        
        # If the action played is illegal, reset
        if not ((a1 in {self.a1_string, self.a2_string}) and (a2 in {self.a1_string, self.a2_string})):
            return obs 
        # If the state tag is not in the observation, simply add the state
        if self.state_tag not in obs:
            return update_state(obs, a1, a2) 

        # Update state counts
        new_obs = self._update_obs_counts(obs, a1, a2, inner_t, outer_t)
        new_obs = update_state(new_obs, a1, a2) if inner_t != 0 else self._remove_state(new_obs)

        return new_obs


class SingleStateOccurrenceUpdater(StateOccurrenceUpdater):
    """Handles state updates and records state occurrence throughout all episodes, without distinguishing between episodes."""

    def __init__(self, a1_string:str, a2_string:str, state_tag: str, additional_info_tag: str, game_description: str, state_prompt: str, instruction: str):
        super().__init__(a1_string, a2_string, state_tag, additional_info_tag, game_description, state_prompt, instruction)

    def _update_obs_counts(self, obs:str, a1:str, a2:str, inner_t:int, outer_t:int) -> str:
        """Returns the updated observation with the provided state counts"""
        current_counts = self._extract_current_counts(obs) # Extract current counts
        a1s, a2s = self._extract_outcome_from_state(obs) # Extract actions from state
        new_counts = self._get_updated_counts(a1s, a2s, current_counts) # Get new counts
        # If starting a new episode, change phrasing to express state count is from a previous game
        replacement = self.current_round_prompt(*new_counts) 
        if "current game" in obs: # IF current game is in the observation, replace pattern
            pattern = self.current_round_prompt(r"\d+", r"\d+", r"\d+", r"\d+")
            return re.sub(pattern, replacement, obs)

        # If current game is not in observation, add the current game prompt before the state
        before_state, after_state = obs.split(self.state_tag)
        separator = "\n" if (self.additional_info_tag in obs) else self.additional_info_tag
        return f"{before_state}{separator}{replacement}{self.state_tag}{after_state}"
    
    def update_obs(self, obs: str, a1:str, a2:str, inner_t:int, outer_t:int, update_state: Callable[[str, str, str], str]) -> str: 
        """Updates the state occurrence based on the input state, and updates the input state based on the new actions."""

        # If the action played is illegal, reset
        if not ((a1 in {self.a1_string, self.a2_string}) and (a2 in {self.a1_string, self.a2_string})):
            return obs
        # If the state tag is not in the observation, simply add the state
        if self.state_tag not in obs:
            return update_state(obs, a1, a2) 
        # Update state counts
        new_obs = self._update_obs_counts(obs, a1, a2, inner_t, outer_t)
        return update_state(new_obs, a1, a2)


class FullTrajectoryUpdater(AdditionalInfoHandler):
    """Handles state updates and records state occurrence throughout the game and the trial."""

    def __init__(self, a1_string:str, a2_string:str, state_tag: str, additional_info_tag: str, game_description: str, state_prompt: str, instruction: str):
        self.state_tag = state_tag
        self.additional_info_tag = additional_info_tag
        self.game_description = game_description
        self.state_prompt = state_prompt
        self.instruction = instruction 

        self.a1_string, self.a2_string = a1_string, a2_string

        self.prev_game = lambda game: f"\nThe trajectory of game {game} was: "
        self.current_game = "\nThe current trajectory of the game has been: "
        self.action_pair = lambda a1, a2: f"{a1}/{a2}, "

    def _format_rounds(self, a1s: List[str], a2s: List[str]) -> str:
        """Given a set of actions, format them into a trajectory according to self.action_pair format."""
        return "".join(self.action_pair(a1, a2) for a1, a2 in zip(a1s, a2s)) # Could be for multiple rounds (in between episodes) 
    
    def _extend_trajectory(self, obs: str, a1s: List[str], a2s: List[str]) -> str:
        """Add new actions to the trajectory in a given observation"""
        before_state, after_state = obs.split(self.state_tag)
        new_rounds = self._format_rounds(a1s, a2s)
        
        # Ensure additional info tag exists
        if self.additional_info_tag not in obs:
            before_state += self.additional_info_tag
            
        if self.current_game in obs: # If a current game trajectory already exists, simply add the new rounds before the state tag. 
            return f"{before_state}{new_rounds}{self.state_tag}{after_state}"
        
        return f"{before_state}{self.current_game}{new_rounds}{self.state_tag}{after_state}" 
    
    def update_obs(self, obs: str, a1:str, a2:str, inner_t:int, outer_t:int, update_state: Callable[[str, str, str], str]) -> str: 
        """Update observation with the actions played in the current round (a1 and a2)."""
        # If the action played is illegal, keep the observation unchanged. 
        if not ((a1 in {self.a1_string, self.a2_string}) and (a2 in {self.a1_string, self.a2_string})):
            return obs

        elif self.state_tag not in obs: # If the observation contains no state, simply add it. This occurs at the beginning of gameplay AND in the first turn of each game (any e, t==1). 
            return update_state(obs, a1, a2) 

        # Extract actions from previous state
        a1s, a2s = self._extract_outcome_from_state(obs) 

        # If the obs is for the next episode, we need to add action from the previous state
        # and the new action taken to the history, instead of using the latter in the state. 
        if inner_t == 0:
            a1s.append(a1)
            a2s.append(a2)

        new_obs = self._extend_trajectory(obs, a1s, a2s) # Extend the trajectory

        # If we are beginning a new episode, replace the current game tag with the previous game one and remove the state 
        if inner_t == 0:
            new_obs = new_obs.replace(self.current_game, self.prev_game(outer_t)) 
            return self._remove_state(new_obs)
        
        return update_state(new_obs, a1, a2)


class ObservationManager:
    """Manages game observations, including state tracking and formatting"""

    # Class constants
    PLACEHOLDERS = {'description': {"a1_tok", "a2_tok", "p1_a1a1", "p1_a1a2", "p1_a2a1", "p1_a2a2", "p2_a1a1", "p2_a1a2", "p2_a2a1", "p2_a2a2"},
        'instruction': {"a1_tok", "a2_tok"}, 'state': {"own_action", "opp_action"}}

    TAGS = {'state': "\n<STATE> ", 'additional_info': "\n<ADDITIONAL INFORMATION> "}

    def __init__(self, config: ObservationManagerConfig):
        
        # Need for formatting and for additional information managers
        self.a1_string, self.a2_string = config.a1_string, config.a2_string
        self.action_string_map = {0: self.a1_string, 1: self.a2_string, 2: "unknown"} 
        self.transmit_info = config.transmit_info

        self.is_shaper = config.is_shaper # Need to reset observation after each episode for non shapers

        # Validate and format prompts
        validate_templates(self.PLACEHOLDERS, [config.game_description, config.instruction_prompt, config.state_prompt])
        # Store game description and instruction and state prompts with the correct legal tokens, reward matrix, and model specific formatting
        self.game_description, self.instruction_prompt, self.state_prompt = self._format_templates(config)

        # Initialize additional information handler
        self.additional_info_handler = self._create_info_handler(config.additional_info_type)


    def _create_info_handler(self, handler_type: str) -> AdditionalInfoHandler:
        """Factory method for creating the appropriate additional information handler"""
        handlers = {
            "state_occurrence": lambda: StateOccurrenceUpdater(
                self.a1_string, self.a2_string,
                self.TAGS['state'],
                self.TAGS['additional_info'],
                self.game_description,
                self.state_prompt,
                self.instruction_prompt
            ),
            "single_state_occurrence": lambda: SingleStateOccurrenceUpdater(
                self.a1_string, self.a2_string,
                self.TAGS['state'],
                self.TAGS['additional_info'],
                self.game_description,
                self.state_prompt,
                self.instruction_prompt
            ),
            "full_trajectory": lambda: FullTrajectoryUpdater(
                self.a1_string, self.a2_string,
                self.TAGS['state'],
                self.TAGS['additional_info'],
                self.game_description,
                self.state_prompt,
                self.instruction_prompt
            ),
            "state_only": lambda: BasicStateUpdater(self.a1_string, self.a2_string)
        }
        
        if handler_type not in handlers:
            raise ValueError(f"Unsupported additional_info_type: {handler_type}")
            
        return handlers[handler_type]()


    def _format_templates(self, config: ObservationManagerConfig) -> None:
        """Formats templates by including the specified allowed tokens, reward matrix, and formatting tags."""
        params = {"a1_tok": self.a1_string, "a2_tok": config.a2_string,
            "p1_a1a1": config.reward_matrix[0][0][0], "p1_a1a2": config.reward_matrix[0][0][1],
            "p1_a2a1": config.reward_matrix[0][1][0], "p1_a2a2": config.reward_matrix[0][1][1],
            "p2_a1a1": config.reward_matrix[1][0][0], "p2_a1a2": config.reward_matrix[1][0][1],
            "p2_a2a1": config.reward_matrix[1][1][0], "p2_a2a2": config.reward_matrix[1][1][1]}

        if config.model_name == "gemma-2b":
            formatted_game_description = config.formatting_tags["start_user_tag"] + (create_payoff_table(params) if (config.game_description == "table") else config.game_description.format(**params))
        else: 
            formatted_game_description = (create_payoff_table(params) if (config.game_description == "table") else config.game_description.format(**params)) + config.formatting_tags["start_user_tag"]

        return (formatted_game_description, 
                config.instruction_prompt.format(**params) + config.formatting_tags["end_tag"] + config.formatting_tags["start_model_tag"], 
                self.TAGS['state'] + config.state_prompt)

    def _format_state(self, a1: str, a2: str) -> str:
        """Format state information with given actions"""
        return self.state_prompt.format(own_action=a1, opp_action=a2)

    def _update_state(self, obs: str, a1: str, a2: str) -> str:
        """Update observation with new state information"""
        if self.TAGS['state'] in obs:
            pattern = self._format_state(r"[A-Za-z]+", r"[A-Za-z]+")
            return re.sub(pattern, self._format_state(a1, a2), obs, count=1)
        
        return obs.replace(self.instruction_prompt, self._format_state(a1, a2) + self.instruction_prompt)


    def batch_update_obs(self, observations: List[str], actions: Tuple[List[str], List[str]], inner_t:int, outer_t:int) -> List[str]:
        """Update multiple observations with their corresponding actions"""
        if not len(observations) == len(actions[0]) == len(actions[1]):
            raise ValueError("Mismatched lengths in batch update inputs")

        if inner_t == 0 and (not self.is_shaper or outer_t == 0 or not self.transmit_info):
            return [self.game_description + self.instruction_prompt] * len(observations)

        return [self.additional_info_handler.update_obs(obs, self.action_string_map[a1], self.action_string_map[a2], inner_t, outer_t, self._update_state) for obs, a1, a2 in zip(observations, actions[0].tolist(), actions[1].tolist())]