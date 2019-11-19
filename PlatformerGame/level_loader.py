import json
import numpy as np
from gamelogic import TileTypes
AIR,BLOCK,START,END,DIAMOND,SPIKE = range(6)
def tiled_tile_value_to_gamelogic_value(tiled_value):
    return {
        0 : TileTypes.AIR,
        1 : TileTypes.BLOCK,
        2 : TileTypes.END,
        3 : TileTypes.DIAMOND,
        4 : TileTypes.START,
        5 : TileTypes.SPIKE,
        6 : TileTypes.AIR,
        7 : TileTypes.BLOCK,
    }[tiled_value]

def load_level(level_file):
    deserialized_level = json.loads(level_file.read())
    game_layer = deserialized_level['layers'][0]
    width = game_layer['width']
    height = game_layer['height']
    level_grid = game_layer['data']
    level_grid = list(map(tiled_tile_value_to_gamelogic_value, level_grid))
    level_grid = np.array(level_grid).reshape(height, width)
    return level_grid.tolist()

def load_level_filepath(level_filepath):
    with open(level_filepath, 'r') as level_file:
        return load_level(level_file)