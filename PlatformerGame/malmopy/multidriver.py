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

"""Driver for experiments with multiple agents on a single thread, sharing an environment"""

import gym

from . import AgentMode, AgentWrapper, Visualizable, VisualizableAgentWrapper, Driver

class StateObservationWrapper(gym.ObservationWrapper, Visualizable):
    """Observation wrapper which exposes the state variable."""

    def __init__(self, env):
        """
        Args:
            env -- the environment to wrap
        """
        super(StateObservationWrapper, self).__init__(env)
        while isinstance(env, gym.Wrapper):
            if hasattr(env, "state"):
                self._state_env = env
                break

            env = env.env

        assert hasattr(env, "state")
        self._state_env = env

    def observation(self, observation):
        return observation

    @property
    def metrics(self):
        return self.env.metrics

    @property
    def state(self):
        """The current state of the environment."""
        return self._state_env.state


class MultiagentWrapper(AgentWrapper):
    """Wrapper enabling this agent to act in a multi-agent scenario"""

    def __init__(self, agent, env):
        agent = VisualizableAgentWrapper(agent)
        super(MultiagentWrapper, self).__init__(agent)
        self._env = StateObservationWrapper(env)
        assert isinstance(env.observation_space, gym.spaces.Tuple)
        assert agent.observation_space == env.observation_space.spaces[1]
        self._pre_observation = None
        self._action = None
        self._reward = None
        self.reset()

    @property
    def env(self):
        """The environment paired to this agent."""
        return self._env

    def observe(self, pre_observation, action, reward, post_observation, done):
        self._pre_observation = pre_observation
        self._action = action
        self._reward = reward

    def act(self, observation):
        if isinstance(observation, tuple):
            observation = observation[1]

        if self._pre_observation is not None:
            self.agent.observe(self._pre_observation, self._action,
                               self._reward, observation, False)

        return self.agent.act(observation)

    def complete_episode(self, post_observation):
        """Completes an episode, providing the final observation.

        Args:
            post_observation -- the terminal observation
        """
        if self.mode != AgentMode.Evaluation:
            self.agent.observe(self._pre_observation, self._action,
                               self._reward, post_observation, True)

        self.reset()

    def reset(self):
        """Resets the internal state of this agent."""
        self._pre_observation = None
        self._action = None
        self._reward = None


class MultiagentDriver(Driver):
    """Driver for multiple agent, turn-based experiments."""

    def __init__(self, agents, params):
        """
        Description:
            Each agent is paired with its own view of the environment, but the first agent
            is assumed to be the primary agent, i.e. reset will be called on their environment,
            they will go first.

        Args:
            agents -- list of MultiagentWrapper wrapped agents and environments
            params -- parameters for the driver (or ArgumentParser)
        """
        for agent in agents:
            assert isinstance(agent, MultiagentWrapper)

        super(MultiagentDriver, self).__init__(agents[0], agents[0].env, params)

        self._agents = list(agents)

    def reset_env(self):
        observation = super(MultiagentDriver, self).reset_env()
        for agent in self._agents:
            agent.reset()

        return observation

    def _set_agent_mode(self, agent):
        if self._train_after == Driver.EVAL_ONLY:
            agent.mode = AgentMode.Evaluation
        elif self._step < self._train_after:
            agent.mode = AgentMode.Warmup
        else:
            agent.mode = AgentMode.Training

    def run(self):
        try:
            turn, _ = self.reset_env()
            while self._stop_trigger(self):
                # loop
                agent = self._agents[turn]
                self._set_agent_mode(agent)

                _, pre_observation = agent.env.state
                action = agent.act(pre_observation)
                (turn, post_observation), reward, done, _ = agent.env.step(action)

                if agent.mode != AgentMode.Evaluation:
                    agent.observe(pre_observation, action, reward, post_observation, done)

                if done:
                    for agent in self._agents:
                        if agent.mode != AgentMode.Evaluation:
                            agent.complete_episode(post_observation)

                    turn, _ = self.reset_env()
                else:
                    for ext in self._extensions:
                        ext(self)

                self._step += 1
        finally:
            for ext in self._extensions:
                ext.finalize()
