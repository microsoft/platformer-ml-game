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

"""Module containing evaluator classes for use in running simple evaluations."""

from typing import List
import logging

from .abc import Evaluator
from .core import AgentMode


class EpisodicEvaluator(Evaluator):
    """
    Evaluate the agent with complete episodes. The length of the episode does not contribute
    to the evaluation, only the final score matters. In case of games with multiple lives, this
    evaluator only use one life.
    """

    def __init__(self, eval_episodes=10, recorder=None):
        """
        Keyword Args:
            eval_episodes -- number of steps in the test [10]
            recorder -- a FileRecorder object [None]
        """
        self._eval_episodes = eval_episodes
        self._logger = logging.getLogger(__name__)
        self._recorder = recorder
        if self._recorder:
            self._recorder.init(('epoch', 'score_per_episode', 'steps_per_episode'))

    def run(self, training_epoch, env, agent) -> list:
        score = 0
        step = 0.0
        agent.mode = AgentMode.Evaluation
        for _ in range(self._eval_episodes):
            # Reset the environment at the beginning of test
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                score += reward
                step += 1

        # Record both the number of epoch and the score
        score_per_episode = score / self._eval_episodes
        steps_per_episode = step / self._eval_episodes
        if self._recorder:
            self._recorder.record((training_epoch, score_per_episode, steps_per_episode))

        self._logger.info("Epoch %d: score_per_episode=%f steps_per_episode=%d", training_epoch,
                          score_per_episode, steps_per_episode)


        return [score_per_episode, steps_per_episode]

    def get_metric_names(self):
        return ["ScorePerEpisode", "StepsPerEpisode"]


class StepEvaluator(Evaluator):
    """
    Evaluate the agent with partial episodes. The reward for each step counts. Usually
    used for maze-liked environment where the time is important.
    """

    def __init__(self, eval_steps=500, recorder=None):
        """
        Keyword Args:
            eval_steps -- number of steps in the test [500]
            recorder -- a FileRecorder object [None]
        """
        self._eval_steps = eval_steps
        self._logger = logging.getLogger(__name__)
        self._recorder = recorder
        self._max_score_per_step = None
        if self._recorder:
            self._recorder.init(('epoch', 'score_per_step'))

    def run(self, training_epoch, env, agent) -> list:
        score = 0

        agent.mode = AgentMode.Evaluation
        current_state = env.reset()
        done = False
        for _ in range(self._eval_steps):
            action = agent.act(current_state)
            current_state, reward, done, _ = env.step(action)
            score += reward

            if done:
                current_state = env.reset()

        # finish up the current episode
        while not done:
            action = agent.act(current_state)
            current_state, _, done, _ = env.step(action)

        # Record both the number of epoch and the score
        score_per_step = score / self._eval_steps
        if self._recorder:
            self._recorder.record((training_epoch, score_per_step))

        if self._max_score_per_step is None:
            self._max_score_per_step = score_per_step
        else:
            self._max_score_per_step = max(score_per_step, self._max_score_per_step)

        self._logger.info("Epoch %d: score_per_step=%f", training_epoch, score_per_step)

        return [score_per_step, self._max_score_per_step]

    def get_metric_names(self) -> List[str]:
        return ["ScorePerStep", "MaxScorePerStep"]
