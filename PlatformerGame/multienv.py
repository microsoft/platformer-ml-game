# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import gym
import os
import json
from PlatformerEnvironment import PlatformerEnvironment
from gamelogic import Controller
from level_loader import load_level_filepath

# replays will be save every REPLAY_SAVE_PERIOD episodes to prevent blocking IO
# killing training time
# This is a prime number to avoid getting stuck on the same level when the number of levels
# is an exact factor of this value
REPLAY_SAVE_PERIOD = 11

class MultiEnv(gym.Env):
    def __init__(
            self,
            levels, 
            max_steps_per_episode, 
            replay_dir=None,
            on_reset_functor=None
        ):
        self._controllers = []
        self._envs = []
        # We start at -1 to counteract the fact that the driver calls reset before we do anything
        # This means we will be bumped up to 0 and start on the first level rather than the second.
        # This makes debugging much more predictable
        self._current_env_index = -1

        self._replay_save_counter = 0
        self._last_replays = None
        self._replay_dir = replay_dir
        self._on_reset_functor = on_reset_functor
        if replay_dir and not os.path.exists(replay_dir):
            os.makedirs(replay_dir)
        
        # Create a controller and PlatformerEnvironment for each level
        for level in levels:
            # Support both file and raw level definitions for testing
            tile_data = load_level_filepath(level) if type(level) is str else level
            print("Level size: %dx%d" % (len(tile_data[0]), len(tile_data)))

            controller = Controller(tile_data)
            env = PlatformerEnvironment(controller, max_steps_per_episode, replay_dir != None)
            self._controllers.append(controller)
            self._envs.append(env)

        self.action_space = self._envs[0].action_space
        self.observation_space = self._envs[0].observation_space

    @property
    def env_count(self):
        return len(self._envs)

    @property
    def combined_level_bounds(self):
        level_bounds_x = 0
        level_bounds_y = 0
        for controller in self._controllers:
            y = len(controller.initial_title_data)
            x = len(controller.initial_title_data[0]) if y > 0 else 0
            level_bounds_x = max(level_bounds_x, x)
            level_bounds_y = max(level_bounds_y, y)
        return level_bounds_x, level_bounds_y
        
    @property
    def diamond_count(self):
        diamond_count = 0
        for env in self._envs:
            diamond_count += env.available_diamond_count
        return diamond_count

    @property
    def last_replays(self):
        return self._last_replays

    @property
    def _current_env(self):
        return self._envs[self._current_env_index]

    @property
    def current_state(self):
        return self._current_env.current_state

    def reset(self):
        self._last_replays = self._current_env.last_first_attempt_replays
        
        if self._last_replays and self._replay_dir:
            self._export_last_replay()
            
        if self._on_reset_functor is not None:
            self._on_reset_functor(has_won = self._current_env.has_first_attempt_win, replay_data = self._last_replays)
        
        self._current_env.reset()
        self._current_env_index += 1
        self._current_env_index %= len(self._envs)

        return self.current_state

    def step(self, action):
        return self._current_env.step(action)

    def render(self, mode='human'):
        raise NotImplementedError

    def _export_last_replay(self):
        self._replay_save_counter += 1
        if self._replay_save_counter % REPLAY_SAVE_PERIOD == 0:
            path = os.path.join(self._replay_dir, "latest.replay")
            with open(path, "w") as f:
                json.dump(self.last_replays, f)

    def close(self):
        for i in range(len(self._envs)):
            env = self._envs[i]
            env.close()
            if (self._replay_dir):
                replay_path = os.path.join(self._replay_dir, str(i) + ".replay")
                env.export_replay_data(replay_path)

    def seed(self, seed=None):
        # It's a fully deterministic environment
        pass