import render
from actionsource import ReplayActionSource
from gamelogic import ActionTypes
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualise a replay.')
    parser.add_argument('replay_file_name', type=str,
                        help='the name of the replay file to load')
    parser.add_argument('--replay_index', type=int, default=0,
                        help='which replay in the file to run')

    args = parser.parse_args()
    
    with open(args.replay_file_name) as f:
        data = json.load(f)
        tile_data = data["tile_data"]
        replays = data["replays"]
    
    action_source = ReplayActionSource(replays[args.replay_index])
    renderer = render.Renderer((len(tile_data[0]), len(tile_data)))
    renderer.render_main(tile_data, [action_source])