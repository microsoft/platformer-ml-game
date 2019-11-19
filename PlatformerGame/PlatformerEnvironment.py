import gym
import numpy as np
import json

# The width/height of the box input to the network
OBSERVATION_WIDTH_IN_SAMPLES = 28
OBSERVATION_HEIGHT_IN_SAMPLES = 20
# The field of view size around the agent
OBSERVATION_WIDTH_IN_TILES = 7
OBSERVATION_HEIGHT_IN_TILES = 5
# Size of each tile so that we can map from world space to tile space
TILE_SIZE = 32

# Tile types
EMPTY = 0
FLOOR = 1
START = 2
EXIT = 3
DIAMOND = 4

# Controller actions
CONTROLLER_LEFT = 1
CONTROLLER_RIGHT = 2
CONTROLLER_JUMPLEFT = 4
CONTROLLER_JUMPRIGHT = 5
# Mapping between agent action space and game action space
ACTION_MAP = (
    CONTROLLER_LEFT,
    CONTROLLER_RIGHT,
    CONTROLLER_JUMPLEFT,
    CONTROLLER_JUMPRIGHT
)

# Observation sizes in the world space
OBSERVATION_WIDTH_IN_PIXELS = OBSERVATION_WIDTH_IN_TILES * TILE_SIZE
OBSERVATION_HEIGHT_IN_PIXELS = OBSERVATION_HEIGHT_IN_TILES * TILE_SIZE
OBSERVATION_STRIDE_X = (OBSERVATION_WIDTH_IN_TILES * TILE_SIZE) // OBSERVATION_WIDTH_IN_SAMPLES
OBSERVATION_STRIDE_Y = (OBSERVATION_HEIGHT_IN_TILES * TILE_SIZE) // OBSERVATION_HEIGHT_IN_SAMPLES

# Velocity limits
# Velocities outside of these limits will be observed as these limits (i.e. clamped)
VELOCITY_LOW = -100
VELOCITY_HIGH = 100


# Samples the world space to create and observation of the required size
class WorldBoxer():
    def __init__(self, tileData, hide_meta_tiles=False):
        self.tileData = tileData
        self.hide_meta_tiles = hide_meta_tiles
        self.height = len(tileData) * TILE_SIZE
        self.width = len(tileData[0]) * TILE_SIZE

    def world_space_to_tile_space(self, y, x):
        return (int(y // TILE_SIZE), int(x // TILE_SIZE))

    def box_around(self, worldY, worldX):
        box = np.ndarray(shape=(OBSERVATION_HEIGHT_IN_SAMPLES, OBSERVATION_WIDTH_IN_SAMPLES), dtype=np.float32)
        initialWorldX = worldX - OBSERVATION_WIDTH_IN_PIXELS // 2
        worldX = initialWorldX
        worldY = worldY - OBSERVATION_HEIGHT_IN_PIXELS // 2

        for y in range(OBSERVATION_HEIGHT_IN_SAMPLES):
            for x in range(OBSERVATION_WIDTH_IN_SAMPLES):
                # If we're trying to see beyond the edge of the world, either fill it with
                # floors or air
                if worldX < 0 or self.width <= worldX:
                    tile = FLOOR
                elif worldY < 0 or self.height <= worldY:
                    tile = EMPTY
                else:
                    tileY, tileX = self.world_space_to_tile_space(worldY, worldX)
                    tile = self.tileData[tileY][tileX]
                    # Hide the start and the diamonds from the agent if requested
                    if self.hide_meta_tiles and (tile == START or tile == DIAMOND):
                        tile = EMPTY

                box[(y, x)] = tile / 4 # 4 = number of types of item, scales all input to between 0 and 1
                worldX += OBSERVATION_STRIDE_X

            worldX = initialWorldX
            worldY += OBSERVATION_STRIDE_Y

        return box


class PlatformerEnvironment(gym.Env):
    def __init__(self, world, max_steps_per_episode, save_replays=False):
        super().__init__()

        # Left, Left+Jump, Right+Jump, Right
        self.action_space = gym.spaces.Discrete(4)
        # Tuple observation spaces don't work 
        # The box is the nearby tiles, and the discrete space is the player's velocity
        tile_observation_sample_count = OBSERVATION_HEIGHT_IN_SAMPLES * OBSERVATION_WIDTH_IN_SAMPLES
        tile_observation_sample_low = np.zeros((tile_observation_sample_count,))
        tile_observation_sample_high = tile_observation_sample_low + 4
        observation_sample_low = np.append(tile_observation_sample_low, np.ones((2,)) * VELOCITY_LOW)
        observation_sample_high = np.append(tile_observation_sample_high, np.ones((2,)) * VELOCITY_HIGH)

        self.observation_space = gym.spaces.Box(low=observation_sample_low, high=observation_sample_high, dtype=np.float32)
        
        self._world = world
        self._world_boxer = WorldBoxer(world.tile_data, True)

        self._last_diamond_count = 0
        self._step_count = 0
        self._max_steps_per_episode = max_steps_per_episode
        self._has_died_this_run = False
        self._has_first_attempt_win = False

        if save_replays:
            self._replays = []
            self._first_attempt_replays = []
            self._current_replay = []
        else:
            self._replays = None
            self._first_attempt_replays = None
            self._current_replay = None

    @property
    def current_state(self):
        pos = self._world.player_pos
        # Tile data is row major, player position is column major
        nearby_tile_data = self._world_boxer.box_around(pos[1], pos[0])
        flat_nearby_tile_data = np.reshape(nearby_tile_data, [OBSERVATION_HEIGHT_IN_SAMPLES * OBSERVATION_WIDTH_IN_SAMPLES])
        velocity_data = np.clip(np.asarray(self._world.player_velocity, dtype=np.float32), VELOCITY_LOW, VELOCITY_HIGH)
        full_observation_space = np.append(flat_nearby_tile_data, velocity_data)
        return full_observation_space
        
    @property
    def has_first_attempt_win(self):
        return self._has_first_attempt_win
        
    @property
    def available_diamond_count(self):
        return self._world.available_diamonds

    def _record_action(self, action):
        if self._current_replay != None:
            self._current_replay.append(action)

    def _end_replay_recording(self):
        if self._current_replay != None and len(self._current_replay) != 0:
            self._replays.append(self._current_replay)
            if not self._has_died_this_run:
                self._first_attempt_replays.append(self._current_replay)
            self._current_replay = []

    def reset(self):
        self._end_replay_recording()
        self._world.reset()
        self._last_diamond_count = 0
        self._step_count = 0
        self._has_died_this_run = False
        self._has_first_attempt_win = False
        return self.current_state

    def step(self, action):
        action = ACTION_MAP[action]
        
        for _ in range(10):
            self._world.step(action)
            self._record_action(action)

        done = False
        # A negative reward for taking a step in order to encourage the agent to move towards
        # the goal as quickly as possible
        reward = -1 

        if self._last_diamond_count < self._world.diamonds:
            # A diamond has been collected, increase the reward to
            # show that the agent is on the right path
            reward = 0
            self._last_diamond_count = self._world.diamonds

        if self._world.finished:
            self._end_replay_recording()
            if self._world.outcome == 0:
                # The exit was reached
                done = True
                reward = 1
                self._has_first_attempt_win = not self._has_died_this_run
            else:
                # Put the agent back to the start
                self._has_died_this_run = True
                self._world.reset()
        
        self._step_count += 1
        if self._step_count >= self._max_steps_per_episode:
            done = True

        return self.current_state, reward, done, None

    def export_replay_data(self, filename):
        root = {}
        root["tile_data"] = self._world.initial_title_data
        root["replays"] = self._replays
        with open(filename, "w") as f:
            json.dump(root, f)

    @property
    def last_replay(self):
        if not self._replays:
            return None

        root = {}
        root["tile_data"] = self._world.initial_title_data
        root["replays"] = [self._replays[-1]]
        return root
        
    @property
    def last_first_attempt_replays(self):
        if not self._first_attempt_replays:
            return None

        root = {}
        root["tile_data"] = self._world.initial_title_data
        root["replays"] = self._first_attempt_replays[-20:]
        return root

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        # Add the current replay to the list
        self._end_replay_recording()

    def seed(self, seed=None):
        # It's a fully deterministic environment
        pass
