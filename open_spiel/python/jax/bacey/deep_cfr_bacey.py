#

#!pip install --upgrade tensorflow jax open_spiel dm-haiku

# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""
import copy
import enum
import sys

from copy import deepcopy

import numpy as np


# import pyspiel


class Action(enum.IntEnum):
    PASS = 0
    BET = 1


class PlayerId(enum.IntEnum):
    CHANCE = -1
    DEFAULT_PLAYER_ID = 0
    INVALID = -3
    MEAN_FIELD = -5
    SIMULTANEOUS = -2
    TERMINAL = -4


# class PrivateInfoType(enum.IntEnum):
#     ALL_PLAYERS = 2
#     NONE = 0
#     SINGLE_PLAYER = 1


class GameType:
    def __init__(
            self,
            short_name,
            long_name,
            dynamics,
            chance_mode,
            information,
            utility,
            reward_model,
            max_num_players,
            min_num_players,
            provides_information_state_string,
            provides_information_state_tensor,
            provides_observation_string,
            provides_observation_tensor,
            provides_factored_observation_string
    ):
        self.short_name = short_name
        self.long_name = long_name
        self.dynamics = dynamics
        self.chance_mode = chance_mode
        self.information = information
        self.utility = utility
        self.reward_model = reward_model
        self.max_num_players = max_num_players
        self.min_num_players = min_num_players
        self.provides_information_state_string = provides_information_state_string
        self.provides_information_state_tensor = provides_information_state_tensor
        self.provides_observation_string = provides_observation_string
        self.provides_observation_tensor = provides_observation_tensor
        self.provides_factored_observation_string = provides_factored_observation_string


_NUM_PLAYERS = 2
_DECK = frozenset([0, 1, 2])


_GAME_TYPE = GameType(
    short_name="python_kuhn_poker",
    long_name="Python Kuhn Poker",
    dynamics=1,  # pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=1,  # pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=2,  # pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=0,  # pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=1,  # pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)

# class GameInfo:
#     def __init__(
#             self,
#             num_distinct_actions,
#             max_chance_outcomes,
#             num_players,
#             min_utility,
#             max_utility,
#             utility_sum,
#             max_game_length
#     ):
#         self.num_distinct_actions = num_distinct_actions
#         self.max_chance_outcomes = max_chance_outcomes
#         self.num_players = num_players
#         self.min_utility = min_utility
#         self.max_utility = max_utility
#         self.utility_sum = utility_sum
#         self.max_game_length = max_game_length

# _GAME_INFO = GameInfo(
#     num_distinct_actions=len(Action),
#     max_chance_outcomes=len(_DECK),
#     num_players=_NUM_PLAYERS,
#     min_utility=-2.0,
#     max_utility=2.0,
#     utility_sum=0.0,
#     max_game_length=3)  # e.g. Pass, Bet, Bet


# class IIGObservationTypeBela:
#     def __init__(self, perfect_recall):
#         self.perfect_recall = perfect_recall


class KuhnPokerGameBela:
    """A Python version of Kuhn poker."""

    # def __init__(self, params=None):
    #     super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return KuhnPokerStateBela(self)

    # def make_py_observer(iig_obs_type=None, params=None):
    #     """Returns an object used for observing game state."""
    #     return KuhnPokerObserverBela(
    #         iig_obs_type or IIGObservationTypeBela(perfect_recall=False),
    #         params)

    @staticmethod
    # @jax.jit
    def num_players():
        return _NUM_PLAYERS

    @staticmethod
    # @jax.jit
    def num_distinct_actions():
        return len(Action)

    @staticmethod
    # @jax.jit
    def get_type():
      return _GAME_TYPE

    def new_initial_states(self):
      return [self.new_initial_state()]



class KuhnPokerStateBela:
    """A python version of the Kuhn poker state."""

    def __init__(self, game):
        """Constructor; should only be called by Game.new_initial_state."""
        # super().__init__(game)
        self.game = game
        self.cards = []
        self.bets = []
        self.pot = [1.0, 1.0]
        self._game_over = False
        self._next_player = 0
        self._history = []

    def current_player(self):
        """Returns id of the next player to move, or TERMINAL if game is over."""
        if self._game_over:
            return PlayerId.TERMINAL
        elif len(self.cards) < self.num_players():
            return PlayerId.CHANCE
        else:
            return self._next_player

    # @jax.jitted
    def legal_actions(self, player=None):
      """Returns a list of legal actions, sorted in ascending order."""

      if not player:
        player = self.current_player()

      if not self.is_terminal() and player == self.current_player():
        if self.is_chance_node():
          return self.legal_chance_outcomes()
        else:
          assert player >= 0, f'player should be >= 0, but it is: {player}'
          return [Action.PASS, Action.BET]
      else:
        logging.error(f'Is it OK to end up here? state: {self} , player: {player}')
        return []

    # @jax.jitted
    def legal_actions_nem_hiszem_hogy_jo(self, player=None):
      """Returns a list of legal actions, sorted in ascending order."""

      if not player:
        player = self.current_player()

      if self.is_terminal() or self.is_chance_node():
        return []
      else:
        assert player >= 0, f'player should be >= 0, but it is: {player}'
        return [Action.PASS, Action.BET]

    def legal_chance_outcomes(self):
      return [action_and_prob[0] for action_and_prob in self.chance_outcomes()]

    def chance_outcomes(self):
        """Returns the possible chance outcomes and their probabilities."""
        assert self.is_chance_node()
        outcomes = sorted(_DECK - set(self.cards))
        p = 1.0 / len(outcomes)
        return [(o, p) for o in outcomes]

    def apply_action(self, action):
        # history_ needs to be modified *after* DoApplyAction which could
        # be using it.

        assert action in Action, f'Received an unknown/invalid action: {action}'

        player = self.current_player()
        self._apply_action(action)
        self._history.append((player, action))

        # And I don't do this:
        #++move_number_;

    def _apply_action(self, action):
        """Applies the specified action to the state."""
        if self.is_chance_node():
            self.cards.append(action)
        else:
            self.bets.append(action)
            if action == Action.BET:
                self.pot[self._next_player] += 1
            self._next_player = 1 - self._next_player
            if ((min(self.pot) == 2) or
                    (len(self.bets) == 2 and action == Action.PASS) or
                    (len(self.bets) == 3)):
                self._game_over = True

    @staticmethod
    def _action_to_string(player, action):
        """Action -> string."""
        if player == PlayerId.CHANCE:
            return f"Deal:{action}"
        elif action == Action.PASS:
            return "Pass"
        else:
            return "Bet"

    def is_terminal(self):
        """Returns True if the game is over."""
        return self._game_over

    def returns(self):
        """Total reward for each player over the course of the game so far."""
        pot = self.pot
        winnings = float(min(pot))
        if not self._game_over:
            return [0., 0.]
        elif pot[0] > pot[1]:
            return [winnings, -winnings]
        elif pot[0] < pot[1]:
            return [-winnings, winnings]
        elif self.cards[0] > self.cards[1]:
            return [winnings, -winnings]
        else:
            return [-winnings, winnings]

    def information_state_tensor(self, player):
        num_players = self.num_players()
        num_cards = len(_DECK)
        num_bet_turns = 3
        num_bet_types = len(Action)

        cards_dealt = self.cards
        current_player = player
        actual_bets = self.bets

        information_state_tensor = np.array([])

        player_encoded = jax.nn.one_hot(current_player, num_players)
        information_state_tensor = jnp.append(
            information_state_tensor,
            player_encoded)

        current_player_has_been_dealt_a_card_already = current_player < len(cards_dealt)

        if current_player_has_been_dealt_a_card_already:
            private_card = cards_dealt[current_player]
            private_card_encoded = jax.nn.one_hot(
                private_card,
                num_cards)
            information_state_tensor = jnp.append(
                information_state_tensor,
                private_card_encoded)
        else:
            information_state_tensor = jnp.append(
                information_state_tensor,
                np.zeros(num_cards, np.float32))

        for current_turn in range(num_bet_turns):

            there_was_a_bet_in_the_current_turn = current_turn < len(actual_bets)

            if there_was_a_bet_in_the_current_turn:
              bet = actual_bets[current_turn]
              bet_encoded = jax.nn.one_hot(bet, num_bet_types)
              information_state_tensor = jnp.append(
                  information_state_tensor,
                  bet_encoded)
            else:
              information_state_tensor = jnp.append(
                  information_state_tensor,
                  np.zeros(num_bet_types, np.float32))

        assert len(information_state_tensor) == num_players + num_cards + \
                                                num_bet_turns * num_bet_types, \
          f'The expected length of the {information_state_tensor} ' \
          f'should be {num_players + num_cards + num_bet_turns * num_bet_types}: ' \
          f'num_players({num_players}) + num_cards({num_cards}) + ' \
          f'num_bet_turns({num_bet_turns}) * num_bet_types({num_bet_types}) ' \
          f'but it is: ({len(information_state_tensor)})'

        return information_state_tensor

    def is_chance_node(self):
        return self.current_player() == PlayerId.CHANCE

    def child(self, action):
        child = copy.deepcopy(self)
        child._apply_action(action)
        return child

    def max_chance_outcomes(self):
        return self.game.num_players() + 1

    def legal_actions_mask(self, player):
        num_actions = self.max_chance_outcomes() if self.is_chance_node() else self.game.num_distinct_actions()
        return [1 if action in self.legal_actions(player) else 0 for action in range(num_actions)]

    def __str__(self):
        """String for debug purposes. No particular semantics are required."""
        return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])

    def apply_action(self):
        raise Exception('Implement self.apply_action')

    def apply_action_with_legality_check(self):
        raise Exception('Implement self.apply_action_with_legality_check')

    def action_to_string(self):
        raise Exception('Implement self.action_to_string')

    def string_to_action(self):
        raise Exception('Implement self.string_to_action')

    def to_string(self):
        raise Exception('Implement self.to_string')

    def is_initial_state(self):
        raise Exception('Implement self.is_initial_state')

    def move_number(self):
        raise Exception('Implement self.move_number')

    def rewards(self):
        raise Exception('Implement self.rewards')

    def player_reward(self):
        raise Exception('Implement self.player_reward')

    def player_return(self, player):
      assert player >= 0
      _returns = self.returns()
      assert player < len(_returns)
      return _returns[player]

    def is_mean_field_node(self):
        raise Exception('Implement self.is_mean_field_node')

    def is_simultaneous_node(self):
        return False

    def is_player_node(self):
        raise Exception('Implement self.is_player_node')

    def history(self):
        return [h[1] for h in self._history]

    def history_str(self):
        return ", ".join(self.history())

    def full_history(self):
        return self._history

    def information_state_string(self, player=None):
        if not player:
          player = self.current_player()

        assert player >= 0
        assert player < self.num_players()
        result = ""

        # Private card
        if len(self.history()) > player:
            result += str(self.history()[player].action)

        # Betting.
        # Perfect recall public info.
        for i in range(self.num_players(), len(self.history())):
            result += 'b' if self.history()[i].action else 'p'

        return result

    def observation_string(self):
        raise Exception('Implement self.observation_string')

    def observation_tensor(self):
        raise Exception('Implement self.observation_tensor')

    def clone(self):
        return deepcopy(self)

    def undo_action(self):
        raise Exception('Implement self.undo_action')

    def apply_actions(self):
        raise Exception('Implement self.apply_actions')

    def apply_actions_with_legality_checks(self):
        raise Exception('Implement self.apply_actions_with_legality_checks')

    def num_distinct_actions(self):
        raise Exception('Implement self.num_distinct_actions')

    def num_players(self):
      return self.game.num_players()

    def get_game(self):
        return self.game

    def get_type(self):
        raise Exception('Implement self.get_type')

    def serialize(self):
        raise Exception('Implement self.serialize')

    def resample_from_infostate(self):
        raise Exception('Implement self.resample_from_infostate')

    def distribution_support(self):
        raise Exception('Implement self.distribution_support')

    def update_distribution(self):
        raise Exception('Implement self.update_distribution')

    def mean_field_population(self):
        raise Exception('Implement self.mean_field_population')



# class KuhnPokerObserverBela:
#     """Observer, conforming to the PyObserver interface (see observation.py)."""
#
#     def __init__(self, iig_obs_type, params):
#         """Initializes an empty observation tensor."""
#         if params:
#             raise ValueError(f"Observation parameters not supported; passed {params}")
#
#         # Determine which observation pieces we want to include.
#         pieces = [("player", 2, (2,))]
#         # if iig_obs_type.private_info == PrivateInfoType.SINGLE_PLAYER:
#         #     pieces.append(("private_card", 3, (3,)))
#         if iig_obs_type.public_info:
#             if iig_obs_type.perfect_recall:
#                 pieces.append(("betting", 6, (3, 2)))
#             else:
#                 pieces.append(("pot_contribution", 2, (2,)))
#
#         # Build the single flat tensor.
#         total_size = sum(size for name, size, shape in pieces)
#         self.tensor = np.zeros(total_size, np.float32)
#
#         # Build the named & reshaped views of the bits of the flat tensor.
#         self.dict = {}
#         index = 0
#         for name, size, shape in pieces:
#             self.dict[name] = self.tensor[index:index + size].reshape(shape)
#             index += size
#
#     def set_from(self, state, player):
#         """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
#         self.tensor.fill(0)
#         if "player" in self.dict:
#             self.dict["player"][player] = 1
#         if "private_card" in self.dict and len(state.cards) > player:
#             self.dict["private_card"][state.cards[player]] = 1
#         if "pot_contribution" in self.dict:
#             self.dict["pot_contribution"][:] = state.pot
#         if "betting" in self.dict:
#             for turn, action in enumerate(state.bets):
#                 self.dict["betting"][turn, action] = 1
#
#     def string_from(self, state, player):
#         """Observation of `state` from the PoV of `player`, as a string."""
#         pieces = []
#         if "player" in self.dict:
#             pieces.append(f"p{player}")
#         if "private_card" in self.dict and len(state.cards) > player:
#             pieces.append(f"card:{state.cards[player]}")
#         if "pot_contribution" in self.dict:
#             pieces.append(f"pot[{int(state.pot[0])} {int(state.pot[1])}]")
#         if "betting" in self.dict and state.bets:
#             pieces.append("".join("pb"[b] for b in state.bets))
#         return " ".join(str(p) for p in pieces)

# Register the game with the OpenSpiel library

# pyspiel.register_game(_GAME_TYPE, KuhnPokerGameBela)


####################

#@title Implements Deep CFR Algorithm.

# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implements Deep CFR Algorithm.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game. To train these networks a reservoir buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.

This implementation uses skip connections as described in the paper if two
consecutive layers of the advantage or policy network have the same number
of units, except for the last connection. Before the last hidden layer
a layer normalization is applied.
"""

import collections
import random

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
# tensorflow is only used for data processing
import tensorflow as tf
import tensorflow_datasets as tfds

# from open_spiel.python import policy
# import pyspiel

# The size of the shuffle buffer used to reshuffle part of the data each
# epoch within one training iteration
ADVANTAGE_TRAIN_SHUFFLE_SIZE = 100000
STRATEGY_TRAIN_SHUFFLE_SIZE = 1000000


# TODO(author3) Refactor into data structures lib.
class ReservoirBufferBela(object):
    """Allows uniform sampling over a stream of data.

    This class supports the storage of arbitrary elements, such as observation
    tensors, integer actions, etc.

    See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
    """

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self._data = []
        self._add_calls = 0

    def add(self, element):
        """Potentially adds `element` to the reservoir buffer.

        Args:
          element: data to be added to the reservoir buffer.
        """
        if len(self._data) < self._reservoir_buffer_capacity:
            self._data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self._data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

        Args:
          num_samples: `int`, number of samples to draw.

        Returns:
          An iterable over `num_samples` random elements of the buffer.

        Raises:
          ValueError: If there are less than `num_samples` elements in the buffer
        """
        if len(self._data) < num_samples:
            raise ValueError('{} elements could not be sampled from size {}'.format(
                num_samples, len(self._data)))
        return random.sample(self._data, num_samples)

    def clear(self):
        self._data = []
        self._add_calls = 0

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    @property
    def data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)


class DeepCFRSolverBela:
    """Implements a solver for the Deep CFR Algorithm.

    See https://arxiv.org/abs/1811.00164.

    Define all networks and sampling buffers/memories.  Derive losses & learning
    steps. Initialize the game state and algorithmic variables.
    """

    def __init__(self,
                 game,
                 policy_network_layers=(256, 256),
                 advantage_network_layers=(128, 128),
                 num_iterations: int = 100,
                 num_traversals: int = 100,
                 learning_rate: float = 1e-3,
                 batch_size_advantage: int = 2048,
                 batch_size_strategy: int = 2048,
                 memory_capacity: int = int(1e6),
                 policy_network_train_steps: int = 5000,
                 advantage_network_train_steps: int = 750,
                 reinitialize_advantage_networks: bool = True):
        """Initialize the Deep CFR algorithm.

        Args:
          game: Open Spiel game.
          policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
          advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
          num_iterations: Number of iterations.
          num_traversals: Number of traversals per iteration.
          learning_rate: Learning rate.
          batch_size_advantage: (int) Batch size to sample from advantage memories.
          batch_size_strategy: (int) Batch size to sample from strategy memories.
          memory_capacity: Number of samples that can be stored in memory.
          policy_network_train_steps: Number of policy network training steps (one
            policy training iteration at the end).
          advantage_network_train_steps: Number of advantage network training steps
            (per iteration).
          reinitialize_advantage_networks: Whether to re-initialize the advantage
            network before training on each iteration.
        """
        all_players = list(range(game.num_players()))
        #super(DeepCFRSolverBela, self).__init__(game, all_players)
        self._game = game
        self._batch_size_advantage = batch_size_advantage
        self._batch_size_strategy = batch_size_strategy
        self._policy_network_train_steps = policy_network_train_steps
        self._advantage_network_train_steps = advantage_network_train_steps
        self._policy_network_layers = policy_network_layers
        self._advantage_network_layers = advantage_network_layers
        self._num_players = game.num_players()
        self._root_node = self._game.new_initial_state()
        self._embedding_size = len(self._root_node.information_state_tensor(0))
        self._num_iterations = num_iterations
        self._num_traversals = num_traversals
        self._reinitialize_advantage_networks = reinitialize_advantage_networks
        self._num_actions = game.num_distinct_actions()
        self._iteration = 1
        self._learning_rate = learning_rate
        self._rngkey = jax.random.PRNGKey(42)

        # Initialize networks
        def base_network(x, layers):
            x = hk.nets.MLP(layers[:-1], activate_final=True)(x)
            x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
            x = hk.Linear(layers[-1])(x)
            x = jax.nn.relu(x)
            x = hk.Linear(self._num_actions)(x)
            return x

        def adv_network(x, mask):
            x = base_network(x, advantage_network_layers)
            x = mask * x
            return x

        def policy_network(x, mask):
            x = base_network(x, policy_network_layers)
            x = jnp.where(mask == 1, x, -10e20)
            x = jax.nn.softmax(x)
            return x

        x, mask = (jnp.ones([1, self._embedding_size]),
                   jnp.ones([1, self._num_actions]))
        self._hk_adv_network = hk.without_apply_rng(hk.transform(adv_network))
        self._params_adv_network = [
            self._hk_adv_network.init(self._next_rng_key(), x, mask)
            for _ in range(self._num_players)
        ]
        self._hk_policy_network = hk.without_apply_rng(hk.transform(policy_network))
        self._params_policy_network = self._hk_policy_network.init(
            self._next_rng_key(), x, mask)

        # initialize losses and grads
        self._adv_loss = optax.l2_loss
        self._policy_loss = optax.l2_loss
        self._adv_grads = jax.value_and_grad(self._loss_adv)
        self._policy_grads = jax.value_and_grad(self._loss_policy)

        # initialize optimizers
        self._opt_adv_init, self._opt_adv_update = optax.adam(learning_rate)
        self._opt_adv_state = [
            self._opt_adv_init(params) for params in self._params_adv_network
        ]
        self._opt_policy_init, self._opt_policy_update = optax.adam(learning_rate)
        self._opt_policy_state = self._opt_policy_init(self._params_policy_network)

        # initialize memories
        self._create_memories(memory_capacity)

        # jit param updates and matched regrets calculations
        self._jitted_matched_regrets = self._get_jitted_matched_regrets()
        self._jitted_adv_update = self._get_jitted_adv_update()
        self._jitted_policy_update = self._get_jitted_policy_update()

    def _get_jitted_adv_update(self):
        """get jitted advantage update function."""

        @jax.jit
        def update(params_adv, opt_state, info_states, samp_regrets, iterations,
                   masks, total_iterations):
            main_loss, grads = self._adv_grads(params_adv, info_states, samp_regrets,
                                               iterations, masks, total_iterations)
            updates, new_opt_state = self._opt_adv_update(grads, opt_state)
            new_params = optax.apply_updates(params_adv, updates)
            return new_params, new_opt_state, main_loss

        return update

    def _get_jitted_policy_update(self):
        """get jitted policy update function."""

        @jax.jit
        def update(params_policy, opt_state, info_states, action_probs, iterations,
                   masks, total_iterations):
            main_loss, grads = self._policy_grads(params_policy, info_states,
                                                  action_probs, iterations, masks,
                                                  total_iterations)
            updates, new_opt_state = self._opt_policy_update(grads, opt_state)
            new_params = optax.apply_updates(params_policy, updates)
            return new_params, new_opt_state, main_loss

        return update

    def _get_jitted_matched_regrets(self):
        """get jitted regret matching function."""

        @jax.jit
        def get_matched_regrets(info_state, legal_actions_mask, params_adv):
            advs = self._hk_adv_network.apply(params_adv, info_state,
                                              legal_actions_mask)
            advantages = jnp.maximum(advs, 0)
            summed_regret = jnp.sum(advantages)
            matched_regrets = jax.lax.cond(
                summed_regret > 0, lambda _: advantages / summed_regret,
                lambda _: jax.nn.one_hot(  # pylint: disable=g-long-lambda
                    jnp.argmax(jnp.where(legal_actions_mask == 1, advs, -10e20)), self
                    ._num_actions), None)
            return advantages, matched_regrets

        return get_matched_regrets

    def _next_rng_key(self):
        """Get the next rng subkey from class rngkey."""
        self._rngkey, subkey = jax.random.split(self._rngkey)
        return subkey

    def _reinitialize_policy_network(self):
        """Reinitalize policy network and optimizer for training."""
        x, mask = (jnp.ones([1, self._embedding_size]),
                   jnp.ones([1, self._num_actions]))
        self._params_policy_network = self._hk_policy_network.init(
            self._next_rng_key(), x, mask)
        self._opt_policy_state = self._opt_policy_init(self._params_policy_network)

    def _reinitialize_advantage_network(self, player):
        """Reinitalize player's advantage network and optimizer for training."""
        x, mask = (jnp.ones([1, self._embedding_size]),
                   jnp.ones([1, self._num_actions]))
        self._params_adv_network[player] = self._hk_adv_network.init(
            self._next_rng_key(), x, mask)
        self._opt_adv_state[player] = self._opt_adv_init(
            self._params_adv_network[player])

    @property
    def advantage_buffers(self):
        return self._advantage_memories

    @property
    def strategy_buffer(self):
        return self._strategy_memories

    def clear_advantage_buffers(self):
        for p in range(self._num_players):
            self._advantage_memories[p].clear()

    def _create_memories(self, memory_capacity):
        """Create memory buffers and associated feature descriptions."""
        self._strategy_memories = ReservoirBufferBela(memory_capacity)
        self._advantage_memories = [
            ReservoirBufferBela(memory_capacity) for _ in range(self._num_players)
        ]
        self._strategy_feature_description = {
            'info_state': tf.io.FixedLenFeature([self._embedding_size], tf.float32),
            'action_probs': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
        }
        self._advantage_feature_description = {
            'info_state': tf.io.FixedLenFeature([self._embedding_size], tf.float32),
            'iteration': tf.io.FixedLenFeature([1], tf.float32),
            'samp_regret': tf.io.FixedLenFeature([self._num_actions], tf.float32),
            'legal_actions': tf.io.FixedLenFeature([self._num_actions], tf.float32)
        }

    def solve(self):
        """Solution logic for Deep CFR."""
        advantage_losses = collections.defaultdict(list)
        for _ in range(self._num_iterations):
            for p in range(self._num_players):
                for _ in range(self._num_traversals):
                    self._traverse_game_tree(self._root_node, p)
                if self._reinitialize_advantage_networks:
                    # Re-initialize advantage network for p and train from scratch.
                    self._reinitialize_advantage_network(p)
                advantage_losses[p].append(self._learn_advantage_network(p))
            self._iteration += 1
        # Train policy network.
        policy_loss = self._learn_strategy_network()
        return None, advantage_losses, policy_loss

    def _serialize_advantage_memory(self, info_state, iteration, samp_regret,
                                    legal_actions_mask):
        """Create serialized example to store an advantage entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'info_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=info_state)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'samp_regret':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=samp_regret)),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _deserialize_advantage_memory(self, serialized):
        """Deserializes a batch of advantage examples for the train step."""
        tups = tf.io.parse_example(serialized, self._advantage_feature_description)
        return (tups['info_state'], tups['samp_regret'], tups['iteration'],
                tups['legal_actions'])

    def _serialize_strategy_memory(self, info_state, iteration,
                                   strategy_action_probs, legal_actions_mask):
        """Create serialized example to store a strategy entry."""
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'info_state':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=info_state)),
                    'action_probs':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=strategy_action_probs)),
                    'iteration':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=[iteration])),
                    'legal_actions':
                        tf.train.Feature(
                            float_list=tf.train.FloatList(value=legal_actions_mask))
                }))
        return example.SerializeToString()

    def _deserialize_strategy_memory(self, serialized):
        """Deserializes a batch of strategy examples for the train step."""
        tups = tf.io.parse_example(serialized, self._strategy_feature_description)
        return (tups['info_state'], tups['action_probs'], tups['iteration'],
                tups['legal_actions'])

    def _add_to_strategy_memory(self, info_state, iteration,
                                strategy_action_probs, legal_actions_mask):
        # pylint: disable=g-doc-args
        """Adds the given strategy data to the memory.

        Uses either a tfrecordsfile on disk if provided, or a reservoir buffer.
        """
        serialized_example = self._serialize_strategy_memory(
            info_state, iteration, strategy_action_probs, legal_actions_mask)
        self._strategy_memories.add(serialized_example)

    def _traverse_game_tree(self, state, player):
        """Performs a traversal of the game tree using external sampling.

        Over a traversal the advantage and strategy memories are populated with
        computed advantage values and matched regrets respectively.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index for this traversal.

        Returns:
          Recursively returns expected payoffs for each action.
        """
        if state.is_terminal():
            # Terminal state get returns.
            return state.returns()[player]
        elif state.is_chance_node():
            # If this is a chance node, sample an action
            chance_outcome, chance_proba = zip(*state.chance_outcomes())
            action = np.random.choice(chance_outcome, p=chance_proba)
            return self._traverse_game_tree(state.child(action), player)
        elif state.current_player() == player:
            # Update the policy over the info set & actions via regret matching.
            _, strategy = self._sample_action_from_advantage(state, player)
            strategy = np.array(strategy)
            exp_payoff = 0 * strategy
            for action in state.legal_actions():
                exp_payoff[action] = self._traverse_game_tree(
                    state.child(action), player)
            ev = np.sum(exp_payoff * strategy)
            samp_regret = (exp_payoff - ev) * state.legal_actions_mask(player)
            self._advantage_memories[player].add(
                self._serialize_advantage_memory(state.information_state_tensor(player),
                                                 self._iteration, samp_regret,
                                                 state.legal_actions_mask(player)))
            return ev
        else:
            other_player = state.current_player()
            _, strategy = self._sample_action_from_advantage(state, other_player)
            # Recompute distribution for numerical errors.
            probs = np.array(strategy)
            probs /= probs.sum()
            sampled_action_index = np.random.choice(range(self._num_actions), p=probs)
            sampled_action = Action(sampled_action_index)
            self._add_to_strategy_memory(
                state.information_state_tensor(other_player), self._iteration, probs,
                state.legal_actions_mask(other_player))
            return self._traverse_game_tree(state.child(sampled_action), player)

    def _sample_action_from_advantage(self, state, player):
        """Returns an info state policy by applying regret-matching.

        Args:
          state: Current OpenSpiel game state.
          player: (int) Player index over which to compute regrets.

        Returns:
          1. (np-array) Advantage values for info state actions indexed by action.
          2. (np-array) Matched regrets, prob for actions indexed by action.
        """
        info_state = jnp.array(
            state.information_state_tensor(player), dtype=jnp.float32)
        legal_actions_mask = jnp.array(
            state.legal_actions_mask(player), dtype=jnp.float32)
        advantages, matched_regrets = self._jitted_matched_regrets(
            info_state, legal_actions_mask, self._params_adv_network[player])
        return advantages, matched_regrets

    def action_probabilities(self, state):
        """Returns action probabilities dict for a single batch."""
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)
        info_state_vector = jnp.array(
            state.information_state_tensor(cur_player), dtype=jnp.float32)
        legal_actions_mask = jnp.array(
            state.legal_actions_mask(cur_player), dtype=jnp.float32)
        probs = self._hk_policy_network.apply(self._params_policy_network,
                                              info_state_vector, legal_actions_mask)
        return {action: probs[action] for action in legal_actions}

    def _get_advantage_dataset(self, player, nr_steps=1):
        """Returns the collected regrets for the given player as a dataset."""
        self._advantage_memories[player].shuffle_data()
        data = tf.data.Dataset.from_tensor_slices(
            self._advantage_memories[player].data)
        data = data.repeat()
        data = data.shuffle(ADVANTAGE_TRAIN_SHUFFLE_SIZE)
        data = data.batch(self._batch_size_advantage)
        data = data.map(self._deserialize_advantage_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        data = data.take(nr_steps)
        return iter(tfds.as_numpy(data))

    def _get_strategy_dataset(self, nr_steps=1):
        """Returns the collected strategy memories as a dataset."""
        self._strategy_memories.shuffle_data()
        data = tf.data.Dataset.from_tensor_slices(self._strategy_memories.data)
        data = data.repeat()
        data = data.shuffle(STRATEGY_TRAIN_SHUFFLE_SIZE)
        data = data.batch(self._batch_size_strategy)
        data = data.map(self._deserialize_strategy_memory)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)
        data = data.take(nr_steps)
        return iter(tfds.as_numpy(data))

    def _loss_adv(self, params_adv, info_states, samp_regrets, iterations, masks,
                  total_iterations):
        """Loss function for our advantage network."""
        preds = self._hk_adv_network.apply(params_adv, info_states, masks)
        loss_values = jnp.mean(self._adv_loss(preds, samp_regrets), axis=-1)
        loss_values = loss_values * iterations * 2 / total_iterations
        return jnp.mean(loss_values)

    def _learn_advantage_network(self, player):
        """Compute the loss on sampled transitions and perform a Q-network update.

        If there are not enough elements in the buffer, no loss is computed and
        `None` is returned instead.

        Args:
          player: (int) player index.

        Returns:
          The average loss over the advantage network of the last batch.
        """
        for data in self._get_advantage_dataset(
                player, self._advantage_network_train_steps):
            (self._params_adv_network[player], self._opt_adv_state[player],
             main_loss) = self._jitted_adv_update(self._params_adv_network[player],
                                                  self._opt_adv_state[player],
                                                  *data, jnp.array(self._iteration))

        return main_loss

    def _loss_policy(self, params_policy, info_states, action_probs, iterations,
                     masks, total_iterations):
        """Loss function for our policy network."""
        preds = self._hk_policy_network.apply(params_policy, info_states, masks)
        loss_values = jnp.mean(self._policy_loss(preds, action_probs), axis=-1)
        loss_values = loss_values * iterations * 2 / total_iterations
        return jnp.mean(loss_values)

    def _learn_strategy_network(self):
        """Compute the loss over the strategy network.

        Returns:
          The average loss obtained on the last training batch of transitions
          or `None`.
        """
        for data in self._get_strategy_dataset(self._policy_network_train_steps):
            (self._params_policy_network, self._opt_policy_state,
             main_loss) = self._jitted_policy_update(self._params_policy_network,
                                                     self._opt_policy_state,
                                                     *data, self._iteration)

        return main_loss


###################

#@title Just a unittest as scratch.

from unittest import TestCase
import unittest
import jax
import jax.numpy as jnp

import numpy as np


class TryTesting(TestCase):

    def t_est_orig(self):
        pieces = [
            ("player", 2, (2,)),
            ("private_card", 3, (3,)),
            ("betting", 6, (3, 2))
        ]

        # Build the single flat tensor.
        total_size = sum(size for name, size, shape in pieces)
        tensor = np.zeros(total_size, np.float32)

        # Build the named & reshaped views of the bits of the flat tensor.
        dict = {}
        index = 0
        for name, size, shape in pieces:
            dict[name] = tensor[index:index + size].reshape(shape)
            index += size

        cards = [0, 1]
        player = 0
        bets = [0, 1]

        dict["player"][player] = 1
        if len(cards) > player:
            dict["private_card"][cards[player]] = 1

        print(f'dict["betting"] before enumerate: {dict["betting"]}')

        for turn, action in enumerate(bets):
            print(f'turn: {turn} , action: {action}')
            print(f'dict["betting"][turn]: {dict["betting"][turn]}')
            print(f'dict["betting"][turn, action]: {dict["betting"][turn, action]}')
            dict["betting"][turn, action] = 1
            print(f'dict["betting"] after setting something to one: {dict["betting"]}')

        print(f'dict["betting"] after enumerate: {dict["betting"]}')

        info_tensor = np.hstack((
            dict["player"],
            dict["private_card"],
            (dict["betting"]).reshape(6)
        ))

        self.assertTrue(np.array_equal(info_tensor, np.array([1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0.])),
                        f'info_tensor: {info_tensor}')

    def t_est_irom1(self):
        num_players = 2
        num_cards = 3
        num_turns = 3

        cards_dealt = [0, 1]
        current_player = 0
        actual_bets = [0, 1]
        num_bet_types = len("pb") # pass, bet

        information_state_tensor = np.array([])

        player_encoded = jax.nn.one_hot(current_player, num_players)
        information_state_tensor = jnp.append(
            information_state_tensor,
            player_encoded)

        current_player_has_been_dealt_a_card_already = current_player < len(cards_dealt)

        if current_player_has_been_dealt_a_card_already:
            private_card = cards_dealt[current_player]
            private_card_encoded = jax.nn.one_hot(
                private_card,
                num_cards)
            information_state_tensor = jnp.append(
                information_state_tensor,
                private_card_encoded)
        else:
            information_state_tensor = jnp.append(
                information_state_tensor,
                np.zeros(num_cards, np.float32))


        for current_turn in range(num_turns):

            there_was_a_bet_in_the_current_turn = current_turn < len(actual_bets)

            if there_was_a_bet_in_the_current_turn:
              bet = actual_bets[current_turn]
              bet_encoded = jax.nn.one_hot(bet, num_bet_types)
              information_state_tensor = jnp.append(
                  information_state_tensor,
                  bet_encoded)
            else:
              information_state_tensor = jnp.append(
                  information_state_tensor,
                  np.zeros(num_bet_types, np.float32))


        self.assertTrue(np.array_equal(information_state_tensor, np.array([1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.])),
                        f'information_state_tensor: {information_state_tensor}')

#unittest.main(argv=[''], verbosity=2, exit=False)


###################

# from absl import app
from absl import logging
import sys

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
# import pyspiel


logging.set_verbosity(logging.INFO)


def main2(unused_argv):
  game_name = "kuhn_poker"
  # logging.info("Loading %s", game_name)
  game = KuhnPokerGameBela()
  deep_cfr_solver = DeepCFRSolverBela(
      game,
      policy_network_layers=(64, 64, 64),
      advantage_network_layers=(64, 64, 64),
      num_iterations=100,
      num_traversals=1000,
      learning_rate=1e-3,
      batch_size_advantage=2048,
      batch_size_strategy=2048,
      memory_capacity=1e7,
      policy_network_train_steps=5000,
      advantage_network_train_steps=750,
      reinitialize_advantage_networks=True)
  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)

  average_policy = policy.tabular_policy_from_callable(
      game, deep_cfr_solver.action_probabilities)

  conv = exploitability.nash_conv(game, average_policy)
  logging.info("Deep CFR in '%s' - NashConv: %s", game_name, conv)

  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  print("Computed player 0 value: {}".format(average_policy_values[0]))
  print("Computed player 1 value: {}".format(average_policy_values[1]))


if __name__ == "__main__":
  main2(None)


###############


import numpy as np

from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import sequence_form_lp
from open_spiel.python.algorithms.get_all_states import get_all_states
from open_spiel.python.games import kuhn_poker  # pylint: disable=unused-import
from open_spiel.python.observation import make_observation
import pyspiel
import pprint

pp = pprint.PrettyPrinter(indent=4)


py_game = pyspiel.load_game("python_kuhn_poker")
obs_types = [None, pyspiel.IIGObservationType(perfect_recall=True)]
py_observations = [make_observation(py_game, o) for o in obs_types]
print('py_observations')
pp.pprint(py_observations)
py_states = get_all_states(py_game)
print('all py_states')
pp.pprint(py_states)

for key, py_state in py_states.items():
  py_state = py_states[key]
  print('py_state')
  print(dir(py_state))
  pp.pprint(py_state)


  print('bets:')
  pp.pprint(py_state.bets)


  print('information_state_string p0:')
  pp.pprint(py_state.information_state_string(0))


  print('information_state_tensor p0:')
  pp.pprint(py_state.information_state_tensor(0))

  print('information_state_string p1:')
  pp.pprint(py_state.information_state_string(1))


  print('information_state_tensor p1:')
  pp.pprint(py_state.information_state_tensor(1))


  print('hist')
  pp.pprint(py_state.history())
  print('returns')
  pp.pprint(py_state.returns())
  for py_obs in py_observations:
    for player in (0, 1):
      py_obs.set_from(py_state, player)
      print('py_obs')
      pp.pprint(py_obs.tensor)



##################

# prompt: Write a jax function which adds a tensor to a global variable. That global variable is array which can hold 10 elements.

import numpy as np
import jax
import jax.numpy as jnp

# Initialize the global variable as a mutable array
global_array = jnp.zeros(10)
#global_array = jax.device_put(global_array) # Put it on the device

@jax.jit
def add_to_global(index, value):
  #global global_array

  global_array.at[index].set(value)
  return global_array

# Example usage
updated_array = add_to_global(5, 5)
print(global_array)

updated_array = add_to_global(3, 3)
print(global_array)


##################


