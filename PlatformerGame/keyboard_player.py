# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import render
from actionsource import KeyboardActionSource
from level_loader import load_level_filepath
import os.path

if __name__ == '__main__':
    level_data = load_level_filepath(os.path.join('Data', 'Test', 'goal_advanced.json'))
    action_source = KeyboardActionSource()
    renderer = render.Renderer((len(level_data[0]), len(level_data)))
    renderer.render_main(level_data, [action_source])