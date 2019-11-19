import pygame
import pygame.locals
import os
import numpy as np
import random
from math import floor
from gamelogic import Controller, TEST_LEVEL, ActionTypes, TileTypes, CharacterState, CharacterDirection
from collections import namedtuple
from actionsource import KeyboardActionSource

def load_tile_table(filename, width, height):
    image = pygame.image.load(filename).convert_alpha()
    image_width, image_height = image.get_size()
    tile_table = []
    for tile_x in range(0, floor(image_width/width)):
        line = []
        tile_table.append(line)
        for tile_y in range(0, floor(image_height/height)):
            rect = (tile_x*width, tile_y*height, width, height)
            line.append(image.subsurface(rect))
    return tile_table

def is_type(map, x, y, type):
    return map[y][x] == type
    
def is_wall(map, x, y):
    return is_type(map, x, y, TileTypes.BLOCK)
    
def is_gem(map, x, y):
    return is_type(map, x, y, TileTypes.DIAMOND)
    
def is_bomb(map, x, y):
    return is_type(map, x, y, TileTypes.SPIKE)

def is_exit(map, x, y):
    return is_type(map, x, y, TileTypes.END)
        
        
MAP_TILE_WIDTH = 64
MAP_TILE_HEIGHT = 64
CHAR_TILE_WIDTH = 96
CHAR_TILE_HEIGHT = 96
CHAR_OFFSET_X = -16
CHAR_OFFSET_Y = -32

FRAME_TICKS = int(1000/60)

SCALING = 1
GAME_LOGIC_TO_GRAPHICS_SCALING = 2

def render_skybox(target):
    background_image = pygame.image.load(os.path.join('data', 'skybox_sideHills.png')).convert()
    image_width, image_height = background_image.get_size()
    target_width, target_height = target.get_size()
    vertical_offset = int((target_height - image_height) / 2)
    number_of_horizontal_repeats = target_width // image_width
    for repeat_index in range(number_of_horizontal_repeats + 1):
        horizontal_offset = repeat_index * image_width
        target.blit(background_image, (horizontal_offset, vertical_offset))

        if vertical_offset > 0:
            top_bar_strip = background_image.subsurface((0,0, image_width,1))
            top_bar_extended = pygame.transform.scale(top_bar_strip, (image_width, vertical_offset))
            target.blit(top_bar_extended, (horizontal_offset,0))
            
            bottom_bar_strip = background_image.subsurface((0,image_height - 1, image_width,1))
            blit_vertical_end = vertical_offset + image_height
            bottom_bar_extended = pygame.transform.scale(bottom_bar_strip, (image_width, target_height - blit_vertical_end))
            target.blit(bottom_bar_extended, (horizontal_offset, blit_vertical_end))
    

def render_background(target, offset, map, tileset):
    wall = lambda x,y: is_wall(map, x, y)
    exit = lambda x,y: is_exit(map, x, y)
    
    for map_y, line in enumerate(map):
        for map_x, c in enumerate(line):
            if wall(map_x, map_y):
                if map_y - 1 >= 0 and wall(map_x, map_y - 1):
                    tile = 6, 0
                else:
                    tile = 0, 0
            elif exit(map_x, map_y):
                tile = 1, 0
            else:
                tile = None
                
            if tile is not None:
                tile_image = tileset[tile[0]][tile[1]]
                target.blit(tile_image,
                           offset_point((map_x*MAP_TILE_WIDTH, map_y*MAP_TILE_HEIGHT), offset))
    
def render_character(screen, position, centering_game_offset, player_image, player_direction, tint):
    flip_dependent_image = pygame.transform.flip(player_image, True, False) if player_direction == CharacterDirection.LEFT else player_image
    tintable_image = flip_dependent_image.copy()
    tintable_image.fill(tint, special_flags=pygame.BLEND_ADD)
    game_relative_render_position = (position[0] * GAME_LOGIC_TO_GRAPHICS_SCALING + CHAR_OFFSET_X, position[1] * GAME_LOGIC_TO_GRAPHICS_SCALING + CHAR_OFFSET_Y)
    screen_relative_render_position = offset_point(game_relative_render_position, centering_game_offset)
    screen.blit(tintable_image, screen_relative_render_position)
    
def offset_point(point, offset):
    return tuple(point_dim + offset_dim for point_dim, offset_dim in zip(point, offset))

Gem = namedtuple('Gem', ['x', 'y'])
Bomb = namedtuple('Bomb', ['x','y'])
def render_gems(screen, centering_game_offset, gems, map, gem_image, collected_image):
    for gem in gems:
        position = offset_point((gem.x * MAP_TILE_WIDTH, gem.y * MAP_TILE_HEIGHT), centering_game_offset)
        if is_gem(map, gem.x, gem.y):
            screen.blit(gem_image, position)
        else:
            screen.blit(collected_image, position)

def render_bombs(screen, centering_game_offset, bombs, map, bomb_image):
    for bomb in bombs:
        position = offset_point((bomb.x * MAP_TILE_WIDTH, bomb.y * MAP_TILE_HEIGHT), centering_game_offset)
        if is_bomb(map, bomb.x, bomb.y):
            screen.blit(bomb_image, position)
            
def find_all_gems(map):
    gems = []
    for row_index in range(map.shape[0]):
        for column_index in range(map.shape[1]):
            if is_gem(map, column_index, row_index):
                gems.append(Gem(x=column_index, y=row_index))
    return gems

def find_all_bombs(map):
    bombs = []
    for row_index in range(map.shape[0]):
        for column_index in range(map.shape[1]):
            if is_bomb(map, column_index, row_index):
                bombs.append(Gem(x=column_index, y=row_index))
    return bombs

character_state_to_image = {
    CharacterState.JUMPING : (1,0),
    CharacterState.FALLING : (1,0),
    CharacterState.LANDED1 : (2,1)
}
def map_character_state_to_image(character_state, is_running, run_frame_index):    
    if character_state == CharacterState.NORMAL:
        if not is_running:
            return (0,0)
        else:
            return (2 + run_frame_index, 0)
    else:
        return character_state_to_image[character_state]

def calculate_centering_game_offset(world_rect, game_rect):
    game_bounds = game_rect.copy()
    game_bounds.center = world_rect.center
    return game_bounds.topleft

class Renderer():
    def __init__(self, maximum_tile_area_to_render):
        pygame.init()
        width = maximum_tile_area_to_render[0] * MAP_TILE_WIDTH
        height = maximum_tile_area_to_render[1] * MAP_TILE_HEIGHT
        
        self._screen_width, self._screen_height = int(width * SCALING), int(height * SCALING)
        self._screen = pygame.display.set_mode((self._screen_width, self._screen_height), pygame.RESIZABLE)

        self._pixel_display = pygame.Surface((width, height))

        self._fixed_background = pygame.Surface((width, height))
        self._dim_borders = pygame.Surface((width, height))
        
        self._tile_table = load_tile_table(os.path.join("data", "tiles.png"), MAP_TILE_WIDTH, MAP_TILE_HEIGHT)
        self._player_images = load_tile_table(os.path.join("data", "player.png"), CHAR_TILE_WIDTH, CHAR_TILE_HEIGHT)

    def _setup_background(self, centering_game_offset, map):
        render_skybox(self._fixed_background)
        render_background(self._fixed_background, centering_game_offset, map, self._tile_table)

    def _setup_dim_borders(self, centering_game_offset, map):
        game_width = map.shape[1] * MAP_TILE_WIDTH
        game_height = map.shape[0] * MAP_TILE_HEIGHT

        game_x_start = centering_game_offset[0]
        game_x_end = game_x_start + game_width

        game_y_start = centering_game_offset[1]
        game_y_end = game_y_start + game_height

        dim_color = (150,150,150)

        self._dim_borders.fill((255,255,255), pygame.Rect(game_x_start,game_y_start, game_x_end,game_y_end))
        self._dim_borders.fill(dim_color, pygame.Rect(0,0, self._dim_borders.get_width(),game_y_start))
        self._dim_borders.fill(dim_color, pygame.Rect(0,game_y_start, game_x_start,game_y_end))
        self._dim_borders.fill(dim_color, pygame.Rect(game_x_end,game_y_start, self._dim_borders.get_width(),game_y_end))
        self._dim_borders.fill(dim_color, pygame.Rect(0,game_y_end, self._dim_borders.get_width(),self._dim_borders.get_height()))

    def render_main(self, level_data, action_sources):
        assert len(action_sources) != 0
        controllers = [Controller(level_data) for _ in action_sources]
        map = np.array(controllers[0].tile_data)
        gems = find_all_gems(map)
        bombs = find_all_bombs(map)
        gem_image = self._tile_table[2][0]
        collected_gem_image = self._tile_table[5][0]
        bomb_image = self._tile_table[4][0]

        game_width = map.shape[1] * MAP_TILE_WIDTH
        game_height = map.shape[0] * MAP_TILE_HEIGHT
        centering_game_offset = calculate_centering_game_offset(self._pixel_display.get_rect(), pygame.Rect(0,0, game_width,game_height))

        self._setup_background(centering_game_offset, map)
        self._setup_dim_borders(centering_game_offset, map)
        player_position = (0, 0)
        is_quitting = False
        last_time = pygame.time.get_ticks()

        accumulated_time = 0
        def is_game_finished(controller, action_source):
            return controller.finished or not action_source.has_more_actions()
        while not all(is_game_finished(controller, action_source) for controller, action_source in zip(controllers, action_sources)) and not is_quitting:
            now_time = pygame.time.get_ticks()
            delta_time = now_time - last_time
            accumulated_time += delta_time
            last_time = now_time
            if accumulated_time > FRAME_TICKS:
                self._pixel_display.blit(self._fixed_background, (0, 0))
                for controller, action_source in zip(controllers, action_sources):
                    # calculate this twice because the next action may have finished it
                    if not is_game_finished(controller, action_source):
                        action = action_source.get_next_action()
                        controller.step(action)
                    has_ended = is_game_finished(controller, action_source)

                    player_position = controller.player_pos
                    player_position = (player_position[0], player_position[1])
                    character_image_index = map_character_state_to_image(controller.character_state, controller.is_running, controller.running_frame)
                    character_tint = (100,100,100) if has_ended else (0,0,0)
                    render_character(self._pixel_display, player_position, centering_game_offset, self._player_images[character_image_index[0]][character_image_index[1]], controller.character_direction, character_tint)
                up_to_date_map = np.array(controllers[0].tile_data)
                render_gems(self._pixel_display, centering_game_offset, gems, up_to_date_map, gem_image, collected_gem_image)
                render_bombs(self._pixel_display, centering_game_offset, bombs, up_to_date_map, bomb_image)
                self._pixel_display.blit(self._dim_borders, (0,0), special_flags = pygame.BLEND_MULT)
                pygame.transform.scale(self._pixel_display, (self._screen_width, self._screen_height), self._screen)
                pygame.display.flip()
                accumulated_time -= FRAME_TICKS
            while pygame.event.peek():
                event = pygame.event.poll()
                if event.type == pygame.locals.QUIT:
                    is_quitting = True
                elif event.type == pygame.VIDEORESIZE:
                    self._screen_width, self._screen_height = event.dict['size']
                    self._screen = pygame.display.set_mode((self._screen_width, self._screen_height), pygame.RESIZABLE)
                else:
                    for action_source in action_sources:
                        action_source.update(event)

                
if __name__ == '__main__':
    print('render.py now contains only code for rendering a scene, use keyboard_player.py to play the game with the keyboard now')