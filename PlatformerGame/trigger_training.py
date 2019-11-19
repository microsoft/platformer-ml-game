import argparse
import os
import multiprocessing
import end_of_game_renderer
import train
from actionsource import ReplayActionSource
import render

REQUEST_REPLAY = 0 # range(1)
NEW_REPLAY, END_OF_GAME_EVENT = range(2)

def run_replay_monitor(renderer_process_connection, combined_level_bounds):
    class IpcReplayClient:
        def __init__(self, connection):
            self._connection = connection
            
            # replay data
            self.tile_data = None
            self.current_replays = None
            
            # end of game data
            self.end_of_game_stats = None
            self.training_level_medal = None
            self.diamond_medal = None
            
        def try_get_next_replay(self):
            renderer_process_connection.send((REQUEST_REPLAY,))
            
            event = renderer_process_connection.recv()
            if event[0] is NEW_REPLAY:
                self.tile_data = event[1]
                self.current_replays = event[2]
                return True
            elif event[0] is END_OF_GAME_EVENT:
                self.end_of_game_stats = event[1]
                self.training_level_medal = event[2]
                self.diamond_medal = event[3]
                return False
                
            return False

    ipc_replay_client = IpcReplayClient(renderer_process_connection)
    
    renderer = render.Renderer(combined_level_bounds)
    while ipc_replay_client.try_get_next_replay():
        action_sources = [ReplayActionSource(replay) for replay in ipc_replay_client.current_replays]
        renderer.render_main(ipc_replay_client.tile_data, action_sources)
    end_of_game_renderer.render_end_of_game_stats(
        ipc_replay_client.end_of_game_stats.games_attempted,
        ipc_replay_client.end_of_game_stats.games_won,
        ipc_replay_client.end_of_game_stats.training_level_count,
        ipc_replay_client.training_level_medal,
        ipc_replay_client.end_of_game_stats.diamond_count,
        ipc_replay_client.diamond_medal
      )
      
def send_replay(connection, replay_data):
    connection.send((NEW_REPLAY, replay_data['tile_data'], replay_data['replays']))
    
def process_events_in_learning_process(connection, replay_data):
    # We need replay_data in order to serve replay requests, without it we will discard the request and enter a deadlock
    if replay_data is not None:
        if connection.poll():
            event = connection.recv()
            if event[0] is REQUEST_REPLAY:
                send_replay(connection, replay_data)

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process location of data folder')
    train.add_to_parser(parser)
    parser.add_argument('--location', type=str, default=r"Data\Train",
                        help='location of training jsons')
    parser.add_argument('--diamonds_medal_thresholds', type=int, nargs=3, required=True,
                        help='the maximum number of diamonds that could achieve each medal (inclusive)')
    parser.add_argument('--levels_medal_thresholds', type=int, nargs=3, required=True,
                        help='the maximum number of diamonds that could achieve each medal (inclusive)')
    return parser

def main(args):
    # an old replay lock might have been left from a crashed/interrupted previous run
    replay_file_path = args.location + "\\latest.replay"
    try:
        os.remove(replay_file_path)
    except FileNotFoundError:
        pass

    # parse all jsons in the location and create the command
    levels = ''
    for file in os.listdir(args.location):
        if file.endswith('.json'):
            levels += os.path.join(args.location, file) + ","
    args.level = levels.rstrip(',')
    args.savereplay = args.location 
    args.savebrain = args.location + '\\agent_brain'
    
    this_process_connection, renderer_process_connection = multiprocessing.Pipe()
    
    def on_reset_functor(has_won, replay_data):
        process_events_in_learning_process(this_process_connection, replay_data)
     
    evaluator = train.PlatformerGameEvaluator(args, on_reset_functor)
    replay_monitor_app = multiprocessing.Process(target=run_replay_monitor, args=(renderer_process_connection, evaluator.combined_level_bounds), name='replay_monitor')
    replay_monitor_app.start()
    eval_results = evaluator.run_training_and_eval()
    this_process_connection.send((END_OF_GAME_EVENT, eval_results, args.levels_medal_thresholds, args.diamonds_medal_thresholds))
    replay_monitor_app.join()

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args)