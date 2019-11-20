# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os

from gamelogic import Controller
from multienv import MultiEnv
from malmopy.agents import QLearnerAgent, RewardMetrics, ExploreExploit
from malmopy.backends.chainer import ChainerQFunction
from malmopy import Explorer, Driver, AgentMode
from malmopy.triggers import IntervalUnit, LimitTrigger
from level_loader import load_level_filepath
from stochastic_evaluation_qlearner_agent import StochasticEvaluationQLearnerAgent

def get_agent(args, env):
    """Create an agent from the arguments and environment"""
    qfunction = ChainerQFunction(env.observation_space, env.action_space, args)
    agent = StochasticEvaluationQLearnerAgent(qfunction, args)
    agent = RewardMetrics("QLearner", ExploreExploit(Explorer.create(args), agent))

    return agent


# TODO - Replace with level loaded from disk
# This sample level's optimal solution takes 112 steps
TRAINING_LEVEL = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,2,0,0,1,0,0,0,3,0],
    [1,1,1,1,0,0,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]
]

TRAINING_SMALL = [
    [0,0,0,0,0],
    [0,2,4,3,0],
    [1,1,1,1,1]
]

MAX_STEPS_PER_EPISODE = 50

def get_environment(args, on_reset_functor):
    levels = []
    if args.level:
        for level in args.level.split(","):
            levels.append(level)

    return MultiEnv(levels, MAX_STEPS_PER_EPISODE, args.savereplay, on_reset_functor)
    
def add_to_parser(arg_parser):
    QLearnerAgent.add_args_to_parser(arg_parser)
    Explorer.add_args_to_parser(arg_parser)
    ChainerQFunction.add_args_to_parser(arg_parser)
    Driver.add_args_to_parser(arg_parser)
    arg_parser.set_defaults(explorer="linear",
                            linear_range=(.9, .05),
                            linear_duration=25000,
                            opt="adam",
                            learning_rate=0.001,
                            discount=0.9,
                            qtype="dqn",
                            model="mlp",
                            hidden_layer_sizes=(512, 512, 512),
                            train_frequency=100,
                            target_update_frequency=1000,
                            minibatch_size=32,
                            memory="replay",
                            num_replay_steps=50000,
                            limit_unit="episode",
                            num_units=100,
                            epoch_length=1000,
                            train_after=500,
                            logdir=os.path.join("results", "PlatformerGame", "dqn"),
                            eval="episodic",
                            eval_freq="epoch",
                            eval_episodes=5,
                            eval_file="rewards.txt",
                            viz="tensorboard")

    arg_parser.add_argument("--level", type=str, default=None,
                           help="Level file to train on")
    arg_parser.add_argument("--goal_level", type=str, default=None,
                           help="Level file to test on")
    arg_parser.add_argument("--savereplay", type=str, default=None,
                           help="Location to save replays")
    arg_parser.add_argument("--savebrain", type=str, default=None,
                           help="Location to save final trained brain to")
    arg_parser.add_argument("--viewer", dest='viewer', action='store_true',
                           help="Display a rendering window")
    arg_parser.add_argument("--training_iterations", type=int, default=100)


def create_parser():
    """Create the argument parser"""
    arg_parser = ArgumentParser(description='Platformer Game MLP DQN example',
                                formatter_class=ArgumentDefaultsHelpFormatter)
    add_to_parser(arg_parser)
    return arg_parser

    
class GoalCounter():
    def __init__(self):
        self._games_completed = 0
        self._games_won = 0
        
    @property
    def games_completed(self):
        return self._games_completed
        
    @property
    def games_won(self):
        return self._games_won
        
    def game_finished(self, is_game_won):
        self._games_completed += 1
        if is_game_won:
            self._games_won += 1

class EvalResults:
    def __init__(self, games_attempted, games_won, training_level_count, diamond_count):
        self.games_attempted = games_attempted
        self.games_won = games_won
        self.training_level_count = training_level_count
        self.diamond_count = diamond_count
        
def nop_on_reset_functor(has_won, replay_data):
    pass
        
class PlatformerGameEvaluator():
    def __init__(self, args, on_reset_functor_arg = nop_on_reset_functor):
        def train_on_reset_functor (has_won, replay_data):
            on_reset_functor_arg(has_won, replay_data)

        self.env = get_environment(args, train_on_reset_functor)
        self.agent = get_agent(args, self.env)
        training_driver_params = Driver.Parameters(args)
        training_driver_params.stop_trigger = LimitTrigger(args.training_iterations * self.env.env_count, IntervalUnit.Episode)
        self.driver = Driver(self.agent, self.env, training_driver_params)
        
        self.goal_counter = GoalCounter()
        def test_on_reset_functor(has_won, replay_data):
            on_reset_functor_arg(has_won, replay_data)
            self.goal_counter.game_finished(has_won)
        self.goal_environment = MultiEnv([args.goal_level], MAX_STEPS_PER_EPISODE, args.savereplay, test_on_reset_functor)
        
        eval_driver_params = Driver.Parameters(args)
        eval_driver_params.stop_trigger = LimitTrigger(100, IntervalUnit.Episode)
        eval_driver_params.train_after = Driver.EVAL_ONLY
        self.eval_driver = Driver(self.agent, self.goal_environment, eval_driver_params)

        self.save_brain = args.savebrain
        
    def run_training_and_eval(self):
        try:
            # Let's run !
            logging.info("Start training...")
            self.driver.run()
            logging.info("Done training.")
            self.eval_driver.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.goal_environment.close()
            self.env.close()
            if self.save_brain:
                print("Saving brain to %s" % self.save_brain)
                self.agent.save(self.save_brain)
        return EvalResults(
            games_attempted = self.goal_counter.games_completed,
            games_won = self.goal_counter.games_won,
            training_level_count = self.env.env_count,
            diamond_count = self.env.diamond_count
        )

    @property
    def combined_level_bounds(self):
        return tuple(max(lhs_dimension, rhs_dimension) for lhs_dimension, rhs_dimension in zip(self.env.combined_level_bounds, self.goal_environment.combined_level_bounds))
        
def main():
    """Main function"""
    logging.basicConfig(level=logging.DEBUG)

    arg_parser = create_parser()

    args = arg_parser.parse_args()
    evaluator = PlatformerGameEvaluator(args)
    evaluator.run_training_and_eval()

if __name__ == '__main__':
    main()
