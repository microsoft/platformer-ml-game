# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from .gamelogic import Controller

class Replayer:
    def __init__(self, reply_file):
        with open(reply_file) as f:
            data = json.load(f)
            self.tile_data = data["tile_data"]
            self._replays = data["replays"]

        self._controllers = []
        for _ in range(len(self._replays)):
            self._controllers.append(Controller(self.tile_data))

        self._step_num = 0
        self.done = False

    def reset(self):
        self._step_num = 0
        self.done = False
        for controller in self._controllers:
            controller.reset()

    def update(self):
        self.done = True

        for i in range(len(self._controllers)):
            if self._step_num < len(self._replays[i]):
                action = self._replays[i][self._step_num]
                self._controllers[i].step(action)
                self.done = False
           
        self._step_num += 1

    def player_positions(self):
        for controller in self._controllers:
            yield controller.player_pos