# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

class TileTypes:
    AIR,BLOCK,START,END,DIAMOND,SPIKE = range(6)

class ActionTypes:
    NOOP, LEFT, RIGHT, JUMP, LEFTJUMP, RIGHTJUMP = range(6)

class OutcomeTypes:
    FOUND_EXIT, SPIKED_TO_DEATH, FELL_OFF_SCREEN = range(3)

class CharacterState:
    NORMAL, JUMPING, FALLING, LANDED1 = range(4)
    
class CharacterDirection:
    RIGHT, LEFT = range(2)

TEST_LEVEL = [
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,2,4,0,0,0,0,4,0,3],
    [1,1,1,1,1,0,0,0,1,1],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,1,0,0,0]
]

GRAVITY = 0.3
TILE_WIDTH = 32
TILE_HEIGHT = 32
WALK_SPEED = 2
JUMP_INITIAL_SPEED = -6
FALL_LIMIT = 100
ANIMATION_SPEED = 6
RUNNING_ANIMATION_SPEED = 10
RUNNING_ANIMATION_FRAMES = 2

class Controller(object):
    def __init__(self, tile_data):
        self._current_pos = None
        for row, row_data in enumerate(tile_data):
            try:
                self._current_pos = (row_data.index(TileTypes.START) * TILE_WIDTH, row * TILE_HEIGHT)
            except ValueError:
                pass
        assert self._current_pos
        self._inital_pos = self._current_pos[:]
        self._inital_tile_data = [] # Explicitly make a copy of the tile data to resore on reset
        for row in tile_data:
            self._inital_tile_data.append(row[:])

        self._tiles_data = tile_data
        self._current_velocity = [0,0]
        self._in_air = False
        self._diamonds = 0
        self._finished = False
        self._fall_start_point = 0
        self._outcome = None
        self._character_state = CharacterState.NORMAL
        self._step_counter = 0
        self._land_timer = 0
        self._character_direction = CharacterDirection.RIGHT

    @property
    def finished(self):
        return self._finished

    @property
    def player_pos(self):
        return self._current_pos

    @property
    def player_velocity(self):
        return self._current_velocity

    @property
    def initial_title_data(self):
        return self._inital_tile_data

    @property
    def tile_data(self):
        return self._tiles_data

    @property
    def character_state(self):
        return self._character_state

    @property
    def outcome(self):
        return self._outcome

    @property
    def diamonds(self):
        return self._diamonds

    @property
    def available_diamonds(self):
        diamonds_found = 0
        for row in self.initial_title_data:
            for cell in row:
                if cell == TileTypes.DIAMOND:
                    diamonds_found += 1
        return diamonds_found
        
    @property
    def character_direction(self):
        return self._character_direction
        
    @property
    def is_running(self):
        return self._current_velocity[0] != 0
        
    @property
    def running_frame(self):
        return self._running_animation_frame

    def reset(self):
        self._current_pos = self._inital_pos[:]
        self._current_velocity = [0,0]
        self._finished = False
        self._outcome = None
        self._in_air = False
        self._fall_start_point = 0
        self._diamonds = 0
        self._step_counter = 0
        self._running_animation_frame = 0
        self._character_state = CharacterState.NORMAL
        # This is an explicit copy of the tile data into the existing arrays rather than a replace
        # so that we don't have to update references to the environment which are held by the environments
        for y in range(len(self._tiles_data)):
            for x in range(len(self._tiles_data[0])):
                self._tiles_data[y][x] = self._inital_tile_data[y][x]

    def step(self, action):
        self._step_counter += 1
        y_vel = self._current_velocity[1]
        old_y_vel = y_vel
        x_vel = WALK_SPEED if action in [ActionTypes.RIGHT, ActionTypes.RIGHTJUMP] else -WALK_SPEED if action in [ActionTypes.LEFT, ActionTypes.LEFTJUMP] else 0
        if x_vel > 0:
            self._character_direction = CharacterDirection.RIGHT
        elif x_vel < 0:
            self._character_direction = CharacterDirection.LEFT
        
        if action in [ActionTypes.JUMP, ActionTypes.LEFTJUMP, ActionTypes.RIGHTJUMP] and not self._in_air:
            self._in_air = True
            y_vel = JUMP_INITIAL_SPEED
            self._character_state = CharacterState.JUMPING
        # Apply acceleration:
        if self._in_air:
            y_vel += GRAVITY
            if y_vel > 0 and old_y_vel <= 0:
                self._fall_start_point = self._current_pos[1]
                self._character_state = CharacterState.FALLING
        self._current_velocity = [x_vel, y_vel]
        self._detect_physics_collisions()
        self._detect_trigger_collisions()
        self._current_pos = (self._current_pos[0] + self._current_velocity[0], self._current_pos[1] + self._current_velocity[1])
        self._check_for_falling()
        if self._character_state >= CharacterState.LANDED1 and self._step_counter % ANIMATION_SPEED == self._land_timer:
            self._character_state = CharacterState.NORMAL
        self._running_animation_frame = (self._step_counter % RUNNING_ANIMATION_SPEED) // (RUNNING_ANIMATION_SPEED // RUNNING_ANIMATION_FRAMES)

    def _snap_x(self):
        x = round(self._current_pos[0] / TILE_WIDTH) * TILE_WIDTH
        self._current_pos = (x, self._current_pos[1])

    def _snap_y(self):
        y = round(self._current_pos[1] / TILE_HEIGHT) * TILE_HEIGHT
        self._current_pos = (self._current_pos[0], y)

    def _check_for_falling(self):
        if not self._in_air:
            xmin = int(self._current_pos[0] // TILE_WIDTH)
            xmax = int((self._current_pos[0] + 0.999 * TILE_WIDTH) // TILE_WIDTH)
            ymin = int(math.ceil(self._current_pos[1] / TILE_HEIGHT))
            if ymin > 0:
                support = False
                for x in range(xmin, xmax + 1):
                    if self._get_tile_type(x, ymin + 1) == TileTypes.BLOCK:
                        support = True
                if not support:
                    self._in_air = True
                    self._character_state = CharacterState.FALLING
                    
    def _get_tile_type(self, x, y):
        if (x >= 0 and x < len(self._tiles_data[0])):
            if (y >= 0 and y < len(self._tiles_data)):
                column = self._tiles_data[y]
                return column[x]
        return TileTypes.AIR    # allow fall off bottom of screen and jump above top
        
    def _check_collide_with_blocking_tile(self, x, y):
        tile = self._get_tile_type(x, y)
        return tile == TileTypes.BLOCK

    def _process_overlap_with_tile(self, x, y):
        tile = self._get_tile_type(x, y)

        if tile == TileTypes.DIAMOND:
            # collect diamond
            self._diamonds += 1
            self._tiles_data[y][x] = TileTypes.AIR
        elif tile == TileTypes.END:
            # end game
            self._outcome = OutcomeTypes.FOUND_EXIT
            self._finished = True
        elif tile == TileTypes.SPIKE:
            self._outcome = OutcomeTypes.SPIKED_TO_DEATH
            self._finished = True


    def _detect_physics_collisions(self):
        left = self._current_pos[0] + self._current_velocity[0] 
        bottom = self._current_pos[1] + self._current_velocity[1]
        right = left + TILE_WIDTH - 0.001
        top = bottom - TILE_HEIGHT + 0.001

        tright = int(right // TILE_WIDTH)
        tleft = int(left // TILE_WIDTH)
        ttop = int(math.ceil(top / TILE_HEIGHT))
        tbottom = int(math.ceil(bottom / TILE_HEIGHT))

        if ttop >= len(self._tiles_data):
            self._finished = True
            self._outcome = OutcomeTypes.FELL_OFF_SCREEN

        if self._current_velocity[0] > 0:
            # can we move right?
            for t in range(ttop, tbottom + 1):
                if self._check_collide_with_blocking_tile(tright, t):
                    self._current_velocity[0] = 0
        if self._current_velocity[0] < 0:
            # can we move left?
            for t in range(ttop, tbottom + 1):
                if self._check_collide_with_blocking_tile(tleft, t):
                    self._current_velocity[0] = 0

        left = self._current_pos[0] + self._current_velocity[0] 
        bottom = self._current_pos[1] + self._current_velocity[1]
        right = left + TILE_WIDTH - 0.001
        top = bottom - TILE_HEIGHT + 0.001
        tright = int(right // TILE_WIDTH)
        tleft = int(left // TILE_WIDTH)

        if self._current_velocity[1] < 0:
            # can we move up?
            for t in range(tleft, tright + 1):
                if self._check_collide_with_blocking_tile(t, ttop):
                    self._current_velocity[1] = 0
        if self._current_velocity[1] > 0:
            # can we move down?
            for t in range(tleft, tright + 1):
                if self._check_collide_with_blocking_tile(t, tbottom):
                    self._current_velocity[1] = 0
                    self._in_air = False
                    self._snap_y()
                    self._character_state = CharacterState.LANDED1
                    self._land_timer = (self._step_counter -1 ) % ANIMATION_SPEED
                    
    def _detect_trigger_collisions(self):
        left = self._current_pos[0] + self._current_velocity[0] 
        bottom = self._current_pos[1] + self._current_velocity[1]
        right = left + TILE_WIDTH - 0.001
        top = bottom - TILE_HEIGHT + 0.001
        
        tright = int(right // TILE_WIDTH)
        tleft = int(left // TILE_WIDTH)
        ttop = int(math.ceil(top / TILE_HEIGHT))
        tbottom = int(math.ceil(bottom / TILE_HEIGHT))
        
        for x in range(tleft, tright + 1):
            for y in range(ttop, tbottom + 1):
                self._process_overlap_with_tile(x, y)

def main():
    controller = Controller(TEST_LEVEL)
    while not controller.finished:
        print(controller._current_pos, controller.diamonds, controller.outcome)
        controller.step(ActionTypes.RIGHTJUMP)
    print(controller._current_pos, controller.diamonds, controller.outcome)

if __name__ == "__main__":
    main()
