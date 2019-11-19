import pygame
import pygame.locals
import pygame.font
import os.path

def draw_9patch_background(target_surface, target_dimensions):
    target_width, target_height = target_dimensions

    background_panel = pygame.image.load(os.path.join('Data', 'green_panel.png')).convert_alpha()
    panel_width, panel_height = background_panel.get_size()
    # corners
    corner_size = 10
    target_surface.blit(background_panel, (0,0), area = (0,0,corner_size,corner_size))
    target_surface.blit(background_panel, (target_width - corner_size,0), area = (panel_width - corner_size,0,corner_size,corner_size))
    target_surface.blit(background_panel, (0,target_height - corner_size), area = (0,panel_height - corner_size,10,10))
    target_surface.blit(background_panel, (target_width - corner_size,target_height - corner_size), area = (panel_width - corner_size,panel_height - corner_size,10,10))
    # edges
    panel_between_corner_width = panel_width - 2 * corner_size
    panel_between_corner_height = panel_height - 2 * corner_size
    target_between_corner_width = target_width - 2 * corner_size
    target_between_corner_height = target_height - 2 * corner_size
    
    top_bar_strip = background_panel.subsurface((corner_size,0, panel_between_corner_width,corner_size))
    top_bar_extended = pygame.transform.scale(top_bar_strip, (target_between_corner_width, corner_size))
    target_surface.blit(top_bar_extended, (corner_size,0))
    
    bottom_bar_strip = background_panel.subsurface((corner_size,panel_height - corner_size, panel_between_corner_width,corner_size))
    bottom_bar_extended = pygame.transform.scale(bottom_bar_strip, (target_between_corner_width, corner_size))
    target_surface.blit(bottom_bar_extended, (corner_size, target_height - corner_size))
    
    left_bar_strip = background_panel.subsurface((0,corner_size, corner_size, panel_between_corner_height))
    left_bar_extended = pygame.transform.scale(left_bar_strip, (corner_size, target_between_corner_height))
    target_surface.blit(left_bar_extended, (0,corner_size))
    
    right_bar_strip = background_panel.subsurface((panel_width - corner_size,corner_size, corner_size,panel_between_corner_height))
    right_bar_extended = pygame.transform.scale(right_bar_strip, (corner_size, target_between_corner_height))
    target_surface.blit(right_bar_extended, (target_width - corner_size, corner_size))
    #center
    center_source = background_panel.subsurface((corner_size,corner_size, panel_between_corner_width, panel_between_corner_height))
    center_stretched = pygame.transform.scale(center_source, (target_between_corner_width,target_between_corner_height))
    target_surface.blit(center_stretched, (corner_size,corner_size))
    
default_font_intensity = 0.15

class Medal:
  NONE, BRONZE, SILVER, GOLD = range(4)
 
def create_text_surface(text, font_size, intensity = default_font_intensity):
    font = pygame.font.Font(os.path.join('Data', 'kenvector_future_thin.ttf'), font_size)
    color = int(255 * intensity)
    return font.render(text, True, (color,color,color), (0,0,0))
    
def create_centered_text(target_surface, target_dimensions, text, y_coord, font_size, blend_mode):
    text_surface = create_text_surface(text, font_size)
    text_width = text_surface.get_width()
    text_left_pad_to_make_center = (target_dimensions[0] - text_width) / 2
    target_surface.blit(text_surface, (text_left_pad_to_make_center,y_coord), special_flags = blend_mode)
    
def create_left_aligned_text(target_surface, target_dimensions, text, y_coord, left_pad, font_size, blend_mode = pygame.BLEND_ADD, intensity = default_font_intensity):
    text_surface = create_text_surface(text, font_size, intensity)
    text_width = text_surface.get_width()
    target_surface.blit(text_surface, (left_pad,y_coord), special_flags = blend_mode)
    
def create_right_aligned_text(target_surface, target_dimensions, text, y_coord, right_pad, font_size, blend_mode = pygame.BLEND_ADD, intensity = default_font_intensity):
    text_surface = create_text_surface(text, font_size, intensity)
    text_width = text_surface.get_width()
    left_pad = target_dimensions[0] - right_pad - text_width
    target_surface.blit(text_surface, (left_pad,y_coord), special_flags = blend_mode)
    
bar_left_margin = 10
bar_right_margin = 75 # the percentage score goes to the right of the bar
    
def draw_empty_bar(target_surface, target_dimensions, y_coord):
    bar_left = pygame.image.load(os.path.join('Data', 'UiAssets', 'barBack_horizontalLeft.png')).convert_alpha()
    bar_mid = pygame.image.load(os.path.join('Data', 'UiAssets', 'barBack_horizontalMid.png')).convert_alpha()
    bar_right = pygame.image.load(os.path.join('Data', 'UiAssets', 'barBack_horizontalRight.png')).convert_alpha()
    
    bar_mid_left = bar_left_margin + bar_left.get_width()
    bar_mid_right = target_dimensions[0] - bar_right_margin - bar_right.get_width()
    stretched_mid = pygame.transform.scale(bar_mid, (bar_mid_right - bar_mid_left, bar_mid.get_height()))
    
    target_surface.blit(bar_left, (bar_left_margin,y_coord))
    target_surface.blit(bar_right, (bar_mid_right,y_coord))
    target_surface.blit(stretched_mid, (bar_mid_left,y_coord))
    
def draw_filled_bar(target_surface, target_dimensions, y_coord, fill_amount=0.5):
    bar_left = pygame.image.load(os.path.join('Data', 'UiAssets', 'barBlue_horizontalLeft.png')).convert_alpha()
    bar_mid = pygame.image.load(os.path.join('Data', 'UiAssets', 'barBlue_horizontalMid.png')).convert_alpha()
    bar_right = pygame.image.load(os.path.join('Data', 'UiAssets', 'barBlue_horizontalRight.png')).convert_alpha()
    
    bar_mid_left = bar_left_margin + bar_left.get_width()
    bar_mid_max_right = target_dimensions[0] - bar_right_margin - bar_right.get_width()
    bar_mid_width = int((bar_mid_max_right - bar_mid_left) * fill_amount)
    bar_mid_right = bar_mid_left + bar_mid_width
    stretched_mid = pygame.transform.scale(bar_mid, (bar_mid_width , bar_mid.get_height()))
    
    target_surface.blit(bar_left, (bar_left_margin,y_coord))
    target_surface.blit(bar_right, (bar_mid_right,y_coord))
    target_surface.blit(stretched_mid, (bar_mid_left,y_coord))
    
def draw_section_title(target_surface, target_dimensions, text, y_coord):
    create_left_aligned_text(target_surface, target_dimensions, text, y_coord, 15, 22, intensity=0.3)
    
def draw_resources_used_text(target_surface, target_dimensions, resources_used, y_coord):
    create_left_aligned_text(target_surface, target_dimensions, '{:0d}'.format(resources_used), y_coord, 15, 50, intensity=0.5)
    
def draw_medal(target_surface, target_dimensions, medal, y_coord):
    filled_diamond = pygame.image.load(os.path.join('Data', 'UiAssets', 'diamond_filled.png')).convert_alpha()
    empty_diamond = pygame.image.load(os.path.join('Data', 'UiAssets', 'diamond_empty.png')).convert_alpha()
    
    assert filled_diamond.get_size() == empty_diamond.get_size()
    
    max_number_of_medals = 3
    x_separation = -16
    x_right_pad = target_dimensions[0] - 25
    x_left_pad = x_right_pad - (filled_diamond.get_width() + x_separation) * (max_number_of_medals - 1) - filled_diamond.get_width()
    for potential_medal in range(max_number_of_medals):
        this_medal_diamond_x_left_pad = x_left_pad + (filled_diamond.get_width() + x_separation) * potential_medal
        this_medal_is_filled = potential_medal < medal
        this_medal_surface = filled_diamond if this_medal_is_filled else empty_diamond
        target_surface.blit(this_medal_surface, (this_medal_diamond_x_left_pad, y_coord))

def setup_end_of_game_screen(target_surface, win_rate = 0.5, levels_used = 1, levels_medal = Medal.NONE, diamonds_used = 15, diamonds_medal = Medal.NONE):
    target_dimensions = target_surface.get_size()
    
    # Background
    draw_9patch_background(target_surface, target_dimensions)
    
    # Title
    create_centered_text(target_surface, target_dimensions, 'Platformer Trainer', 15, 15, pygame.BLEND_ADD)
    create_centered_text(target_surface, target_dimensions, 'Results', 20, 45, pygame.BLEND_SUB)
    
    # Win rate
    draw_section_title(target_surface, target_dimensions, 'Win Rate', 70)
    draw_empty_bar(target_surface, target_dimensions, 100)
    draw_filled_bar(target_surface, target_dimensions, 100, win_rate)
    create_right_aligned_text(target_surface, target_dimensions, '{:3.0f}%'.format(win_rate * 100), 98, 15, 18, intensity=0.5)
    
    # Levels
    draw_section_title(target_surface, target_dimensions, 'Levels Used', 130)
    draw_resources_used_text(target_surface, target_dimensions, levels_used, 160)
    draw_medal(target_surface, target_dimensions, levels_medal, 160)
    
    # Diamonds
    draw_section_title(target_surface, target_dimensions, 'Diamonds Used', 260)
    draw_resources_used_text(target_surface, target_dimensions, diamonds_used, 290)
    draw_medal(target_surface, target_dimensions, diamonds_medal, 290)

def render_end_of_game_stats(
      goal_level_attempts,
      goal_level_wins,
      levels_used,
      levels_threshold,
      diamonds_used,
      diamonds_threshold
  ):
    pygame.init()
    end_of_game_display = pygame.display.set_mode((320, 380))
    win_rate = goal_level_wins / goal_level_attempts
    
    def find_medal_for_value_and_thresholds(resource_value, resource_thresholds):
        # medals are an enum from NONE = 0 to GOLD = 3
        medal_value = (next( (index for index, threshold in enumerate(resource_thresholds) if threshold >= resource_value), 3))
        return 3 - medal_value
    
    level_medal = find_medal_for_value_and_thresholds(levels_used, levels_threshold)
    diamonds_medal = find_medal_for_value_and_thresholds(diamonds_used, diamonds_threshold)
    
    setup_end_of_game_screen(end_of_game_display, win_rate, levels_used = levels_used, levels_medal = level_medal, diamonds_used = diamonds_used, diamonds_medal = diamonds_medal)
    pygame.display.flip()
    is_quitting = False
    while not is_quitting:
        while pygame.event.peek():
            event = pygame.event.poll()
            if event.type == pygame.locals.QUIT:
                is_quitting = True
                
def main():
    # test the end of game stats with different values
    levels_threshold = [3,5,7]
    diamonds_threshold = [5,10,15]
    render_end_of_game_stats(100, 0, 10, levels_threshold, 5, diamonds_threshold)
    render_end_of_game_stats(100, 50, 5, levels_threshold, 15, diamonds_threshold)
    render_end_of_game_stats(100, 100, 1, levels_threshold, 90, diamonds_threshold)
    
if __name__ == '__main__':
    main()