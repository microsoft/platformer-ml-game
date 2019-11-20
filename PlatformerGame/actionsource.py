# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pygame
import pygame.locals
from gamelogic import ActionTypes

class ActionSource:
    def __init__(self):
        pass

    def get_next_action(self):
        pass

    def update(self, event):
        pass

    def has_more_actions(self):
        return False

class KeyboardActionSource(ActionSource):
    def __init__(self):
        super().__init__()
        self._key_left = False
        self._key_right = False
        self._key_jump = False

    def get_next_action(self):
        if self._key_jump:
            action = ActionTypes.LEFTJUMP if (self._key_left and not self._key_right) else ActionTypes.RIGHTJUMP if (self._key_right and not self._key_left) else ActionTypes.JUMP
        else:
            action = ActionTypes.LEFT if (self._key_left and not self._key_right) else ActionTypes.RIGHT if (self._key_right and not self._key_left) else ActionTypes.NOOP
        return action

    def update(self, event):
        if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
            if event.key == pygame.K_d:
                self._key_right = event.type == pygame.KEYDOWN
            elif event.key == pygame.K_a:
                self._key_left = event.type == pygame.KEYDOWN
            elif event.key == pygame.K_SPACE:
                self._key_jump = event.type == pygame.KEYDOWN

    def has_more_actions(self):
        return True


class ReplayActionSource(ActionSource):
    def __init__(self, action_queue):
        super().__init__()
        self.action_index = 0
        self.action_queue = action_queue

    def get_next_action(self):
        assert self.action_index < len(self.action_queue)
        action_value = self.action_queue[self.action_index]
        self.action_index = self.action_index + 1
        return action_value
    
    def update(self, event):
        pass    # not needed

    def has_more_actions(self):
        return self.action_index < len(self.action_queue)
