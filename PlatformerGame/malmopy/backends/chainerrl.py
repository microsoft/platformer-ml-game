# --------------------------------------------------------------------------------------------------
#  Copyright (c) 2018 Microsoft Corporation
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#  associated documentation files (the "Software"), to deal in the Software without restriction,
#  including without limitation the rights to use, copy, modify, merge, publish, distribute,
#  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or
#  substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------------------------------

"""Wrapper classes for the ChainerRL backend."""

from argparse import Namespace

from chainerrl.replay_buffer import AbstractReplayBuffer
import gym

from .. import Agent, AgentMode, Visualizable, Driver
from ..utils import np_random


class ChainerRLAgent(Agent, Visualizable):
    """Wrapper class for a ChainerRL Agent."""

    def __init__(self, agent, observation_space, action_space):
        super(ChainerRLAgent, self).__init__(observation_space, action_space)
        self._agent = agent
        self._last_reward = 0

    @property
    def metrics(self):
        return []

    def act(self, observation):
        if self.mode == AgentMode.Training:
            return self._agent.act_and_train(observation, self._last_reward)

        return self._agent.act(observation)

    def observe(self, pre_observation, action, reward, post_observation, done):
        self._last_reward = reward

    def stop_episode(self, observation, reward, done=False):
        """Stops an episode.

        Args:
            observation -- the final observation
            reward -- the final reward

        Keyword Args:
            done -- whether the episode is complete as of this method being called [False]
        """
        if self.mode == AgentMode.Training:
            self._agent.stop_episode_and_train(observation, reward, done)
        else:
            self._agent.stop_episode()

    def save(self, path):
        self._agent.save(path)

    def load(self, path):
        self._agent.load(path)

class ChainerRLDriver(Driver):
    """
    This class implements the experiments.train_agent function of ChainerRL.
    """
    EVAL_ONLY = -1

    def __init__(self, agent: ChainerRLAgent, environment: gym.Env, params, step_hooks=None):
        if isinstance(params, Namespace):
            params = Driver.Parameters(params)

        super(ChainerRLDriver, self).__init__(agent, environment, params)

        self._step_hooks = [] if step_hooks is None else step_hooks

    def _loop_step(self, current_state):
        agent, env = self.agent, self.environment

        agent.mode = AgentMode.Warmup if self._step < self._train_after else AgentMode.Training
        action = agent.act(current_state)

        new_state, reward, done, _ = env.step(action)
        agent.observe(current_state, action, reward, new_state, done)

        for hook in self._step_hooks:
            hook(env, agent, self.current_step)

        if done:
            agent.unwrapped.stop_episode(new_state, reward, done=done)
            # Start a new episode
            return self.reset_env()

        return new_state


class ChainerRLMemory(AbstractReplayBuffer):
    """Wrapper which makes a Memory object look like an AbstractReplayBuffer"""

    def __init__(self, memory):
        self._memory = memory
        self.seed()

    def seed(self, seed=None):
        """Seed the PRNG for this memory (used for sampling).

        Keyword Args:
            seed -- the random seed to use [None]

        Returns the random seed used (i.e. if one was generated automatically)
        """
        self.np_random, seed = np_random(seed)
        return seed

    def append(self, state, action, reward, next_state=None, next_action=None,
               is_state_terminal=False):
        #pylint: disable=too-many-arguments
        self._memory.append(state, action, reward, is_state_terminal)

    def sample(self, n):
        batch = self._memory.minibatch(self.np_random, n)
        return batch.to_dict_list()

    def __len__(self):
        return len(self._memory)

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def stop_current_episode(self):
        """This seems to be required due to some oddness within ChainerRL."""
        pass
