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

"""Module containing agent wrappers."""

import numpy as np
import gym

from .. import AgentMode, ActWrapper, ObserveWrapper, Visualizable, Explorer, Agent, SumSummary, \
    MinSummary, MaxSummary, each_episode
from .core import MultiSimultaneousAgent


class RewardMetrics(ObserveWrapper, Visualizable):
    """Class which adds summary metrics related to rewards"""

    def __init__(self, name, agent):
        """
        Args:
            name -- the name of the agent (used in metric tags)
            agent -- the agent to wrap
        """
        super(RewardMetrics, self).__init__(agent)
        self._sum_rewards_summary = SumSummary(name + '/Sum_Rewards', trigger=each_episode())
        self._min_rewards_summary = MinSummary(name + '/Min_Rewards', trigger=each_episode())
        self._max_rewards_summary = MaxSummary(name + '/Max_Rewards', trigger=each_episode())

    @property
    def metrics(self):
        return [self._sum_rewards_summary, self._min_rewards_summary, self._max_rewards_summary]

    def observe(self, pre_observation, action, reward, post_observation, done):
        self._sum_rewards_summary.add(reward)
        self._min_rewards_summary.add(reward)
        self._max_rewards_summary.add(reward)
        self.agent.observe(pre_observation, action, reward, post_observation, done)


class ExploreExploit(ActWrapper, Visualizable):
    """Wrapper which implements an explore/exploit strategy on top of an agent."""

    def __init__(self, explorer, agent):
        assert isinstance(explorer, Explorer)
        super(ExploreExploit, self).__init__(agent)
        self._explorer = explorer
        self._steps = 0

    @property
    def metrics(self):
        if isinstance(self._explorer, Visualizable):
            return self._explorer.metrics

        return []

    def act(self, observation):
        if self.mode == AgentMode.Evaluation:
            action = self.agent.act(observation)
        else:
            self._steps += 1
            if self._explorer.is_exploring(self._steps):
                action = self._explorer.explore(self._steps, self.action_space)
            else:
                action = self.agent.act(observation)

        return action


class RandomReward(ObserveWrapper):
    """Wrapper which models random rewards, passing the expected value to the wrapped agent."""

    def __init__(self, agent):
        super(RandomReward, self).__init__(agent)
        assert isinstance(agent.observation_space, gym.spaces.Discrete)
        assert isinstance(agent.action_space, gym.spaces.Discrete)

        self._means = np.zeros((agent.observation_space.n, agent.action_space.n), np.float64)
        self._counts = np.zeros((agent.observation_space.n, agent.action_space.n), np.int64)

    def observe(self, pre_observation, action, reward, post_observation, done):
        self._counts[pre_observation, action] += 1
        delta = reward - self._means[pre_observation, action]
        self._means[pre_observation, action] += delta / self._counts[pre_observation, action]
        reward = np.asscalar(self._means[pre_observation, action].astype(np.float32))
        self.agent.observe(pre_observation, action, reward, post_observation, done)


class MultiSimultaneousAgentWrapper(ObserveWrapper, MultiSimultaneousAgent):
    """Makes any (single) agent algorithm support simultaneous multi-agent.
       It currently works by giving the agent only his part of the reward
       And making him observe only his part of action.
    """

    def __init__(self, agent):
        self._agent = agent
        ObserveWrapper.__init__(self, self._agent)
        MultiSimultaneousAgent.__init__(self)

    def observe(self, pre_observation, action, reward, post_observation, done):
        return self._agent.observe(pre_observation,
                                   action[self.multi_agent_id],
                                   reward[self.multi_agent_id],
                                   post_observation,
                                   done)


class MultiSimultaneousAgentController(Agent, Visualizable):
    """This is a controller agent that takes multiple agents as input that will play
    simultaneously.
    """

    def __init__(self, agents):
        self._agents = agents
        if not agents:
            raise ValueError("No agents was provided. You need to provide at least 1 agent.")
        # Initialize with the observation and action space
        action_space = gym.spaces.Tuple([agent.action_space for agent in self._agents])
        super().__init__(self._agents[0].observation_space, action_space)

        # If the agent does not natively support multi agent,
        # wrap it into a multi agent class
        for agent_id, agent in enumerate(self._agents):
            if not isinstance(agent, MultiSimultaneousAgent):
                self._agents[agent_id] = MultiSimultaneousAgentWrapper(agent)

            # Set the id of this agent
            self._agents[agent_id].multi_agent_id = agent_id

    def act(self, observation):
        # Get all the action from each agent to be forwarded by the driver to the environment
        action = tuple([agent.act(observation) for agent in self._agents])

        return action

    def observe(self, pre_observation, action, reward, post_observation, done):

        # Full-Information
        # Send the rewards and actions to all agents
        # Each agent can access his own action/reward using his id.
        for agent in self._agents:
            agent.observe(pre_observation, action, reward, post_observation, done)

    @property
    def metrics(self):
        metrics = []
        for agent in self._agents:
            if agent.metrics:
                metrics.extend(agent.metrics)

        return metrics

    def save(self, path):
        for agent_id, _ in enumerate(self._agents):
            self._agents[agent_id].save(path + '_' + self._agents[agent_id].name)

    def load(self, path):
        for agent_id, _ in enumerate(self._agents):
            self._agents[agent_id].load(path + '_' + self._agents[agent_id].name)

    @property
    def mode(self):
        return self._agents[0].mode

    @mode.setter
    def mode(self, value):
        for agent in self._agents:
            agent.mode = value
