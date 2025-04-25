import sys
from PIL import Image
import cv2
import numpy as np
import pygame
from scipy.spatial import Delaunay, ConvexHull
import random

random.seed(10)

# Function to get the bounding rectangle of a polygon
def get_bounding_rect(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min)

# Function to calculate the heuristic for A* algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Function to reconstruct the path from the A* algorithm
def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# A* algorithm implementation
def a_star_with_path(start, goal, obstacles):
    open_set = {start}
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
        if goal[0] - 15 <= current[0] <= goal[0] + 15 and goal[1] - 15 <= current[1] <= goal[1] + 15:
            return reconstruct_path(came_from, current)

        open_set.remove(current)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < 512 and 0 <= neighbor[1] < 512:
                if any(get_bounding_rect(obstacle).collidepoint(neighbor) for obstacle in obstacles):
                    continue

                tentative_g_score = g_score[current] + (1.414 if dx != 0 and dy != 0 else 1)
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    open_set.add(neighbor)
    return None



# Function to control the player movement
def controller(player_speed=5):
    dx, dy = 0, 0
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        dy = -player_speed
    if keys[pygame.K_DOWN]:
        dy = player_speed
    if keys[pygame.K_LEFT]:
        dx = -player_speed
    if keys[pygame.K_RIGHT]:
        dx = player_speed
    return dx, dy


# Function to calculate the Minkowski sum of an obstacle and player size
def minkowski_sum(obstacle, player_size):
    player_width, player_height = player_size[0] + 2, player_size[1] + 2
    player_shape = [
        (-player_width/2, -player_height/2),
        (-player_width/2, player_height/2),
        (player_width/2, player_height/2),
        (player_width/2, -player_height/2)
    ]
    expanded_points = []
    for ox, oy in obstacle:
        for px, py in player_shape:
            expanded_points.append((ox + px, oy + py))
    hull = ConvexHull(expanded_points)
    return [tuple(hull.points[vertex]) for vertex in hull.vertices]

# Function to count the contours of asteroids in the image
def counturing(img_array):
    ast_obstacles = []
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0, True)
        if len(approx) > 2:
            ast_obstacle = [tuple(point[0]) for point in approx]
            ast_obstacles.append(ast_obstacle)
    return ast_obstacles

# Function to sample free space points in the image
def free_space_point_sampler(ast_obstacles):
    free_space_points = []
    for y in range(0, 512, 20):
        for x in range(0, 512, 20):
            point = (x, y)
            if not any(get_bounding_rect(ast_obstacles).collidepoint(point) for ast_obstacles in ast_obstacles):
                free_space_points.append(point)
    free_space_points = random.sample(free_space_points, len(free_space_points) // 2)
    return free_space_points


def main():
    if len(sys.argv) < 5:
        print("Usage: python main.py <input_image_path> <visualize (true/false)> <lava_count> <motion_plan>")
        sys.exit(1)

    input_image_path = sys.argv[1]
    visualize = sys.argv[2].lower() == "true"
    lava_count = int(sys.argv[3])
    motion_plan = sys.argv[4].lower() == "true"

    def reset_game():
        pygame.quit()
        main()


    player_size = (15, 30)

    try:
        with Image.open(input_image_path) as img:
            resized_img = img.resize((512, 512))
            img_array = np.array(resized_img)
            
            pygame.init()
            screen = pygame.display.set_mode((512, 512))
            pygame.display.set_caption("Game with ast_obstacles")
            clock = pygame.time.Clock()
            asteroid_texture = pygame.image.load("asteroid.png").convert_alpha()
            background_image = pygame.image.load("space.png").convert()
            background_image = pygame.transform.scale(background_image, (512, 512))
            
            ast_obstacles = counturing(img_array)
           
            player_speed = 5
            
            free_space_points = free_space_point_sampler(ast_obstacles)

            if len(free_space_points) < 4:
                print("Error: Not enough free space for Delaunay triangulation.")
                pygame.quit()
                sys.exit(1)


            # Perform Delaunay triangulation on the free space points
            triangulation = Delaunay(free_space_points)
            triangle_colors = [
                (
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    np.random.randint(50, 255),
                    100
                )
                for _ in range(len(triangulation.simplices))
            ]

            lava_triangles = random.sample(list(triangulation.simplices), min(lava_count, len(triangulation.simplices)))
            rocket_image = pygame.image.load("rocket.png").convert_alpha()
            rocket_image = pygame.transform.scale(rocket_image, (30, 30))

           
           
            # Create a mask for the asteroids
            expanded_obstacles = []

            for ast_obstacle in ast_obstacles:
                expanded_obstacle = minkowski_sum(ast_obstacle, player_size)
                expanded_obstacles.append(expanded_obstacle)

            # Create a mask for the lava triangles
            expanded_lava_triangles = []

            for simplex in lava_triangles:
                triangle = [free_space_points[i] for i in simplex]
                expanded_lava = minkowski_sum(triangle, player_size)
                expanded_lava_triangles.append(expanded_lava)

            free_space_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
            free_space_surface.fill((0, 0, 255, 50))
            bright_rocket = rocket_image.copy()
            bright_rocket.fill((50, 50, 50, 0), special_flags=pygame.BLEND_RGBA_ADD)
            lava_texture = pygame.image.load("lava.png").convert_alpha()
            valid_triangles = []
            
            # Check for valid triangles that do not intersect with asteroids or lava
            for simplex in triangulation.simplices:
                triangle = [free_space_points[i] for i in simplex]
                triangle_rect = get_bounding_rect(triangle)
                if not any(triangle_rect.colliderect(get_bounding_rect(ast_obstacle)) for ast_obstacle in ast_obstacles) and not any(np.array_equal(simplex, lava) for lava in lava_triangles):
                    valid_triangles.append(simplex)

            if not valid_triangles:
                print("Error: No valid triangles available for the player or objective.")
                pygame.quit()
                sys.exit(1)

            # Randomly select a triangle for the player and objective
            valid_player_position = False
            while not valid_player_position:
                player_triangle = random.choice(valid_triangles)
                player_position = np.mean([free_space_points[i] for i in player_triangle], axis=0).astype(int)
                player = pygame.Rect(player_position[0], player_position[1], 13, 28)
                if not any(get_bounding_rect(obstacle).colliderect(player) for obstacle in expanded_obstacles) and \
                   not any(get_bounding_rect(lava).colliderect(player) for lava in expanded_lava_triangles):
                    valid_player_position = True

            initial_player_position = player.copy()
            valid_objective_position = False

            # Randomly select a triangle for the objective
            while not valid_objective_position:
                objective_triangle = random.choice(valid_triangles)
                objective_position = np.mean([free_space_points[i] for i in objective_triangle], axis=0).astype(int)
                objective_rect = pygame.Rect(objective_position[0] - 15, objective_position[1] - 15, 30, 30)
                if not any(get_bounding_rect(obstacle).colliderect(objective_rect) for obstacle in expanded_obstacles) and \
                   not any(get_bounding_rect(lava).colliderect(objective_rect) for lava in expanded_lava_triangles):
                    valid_objective_position = True

            objective_texture = pygame.image.load("objective.png").convert_alpha()
            objective_texture = pygame.transform.scale(objective_texture, (30, 30))
            player_position = (player.centerx, player.centery)
            objective_position = (objective_rect.centerx, objective_rect.centery)
            all_obstacles = expanded_obstacles + expanded_lava_triangles
            path = a_star_with_path(player_position, objective_position, all_obstacles)

            if path:
                print("Path exists between the player and the objective.")
                if motion_plan:
                    print("Motion plan enabled. Player will start moving in 3 seconds.")
                    pygame.time.wait(3000)
            else:
                print("No path exists between the player and the objective.")
            lava_masks = []
            for simplex in lava_triangles:
                triangle = [free_space_points[i] for i in simplex]
                lava_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
                pygame.draw.polygon(lava_surface, (255, 255, 255), triangle)
                lava_masks.append(pygame.mask.from_surface(lava_surface))

            reset_button = pygame.Rect(450, 10, 50, 30)
            running = True
            game_won = False
            path_index = 0

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if reset_button.collidepoint(event.pos):
                            reset_game()
                        if game_won and restart_button.collidepoint(event.pos):
                            reset_game()
                if not game_won:
                    if motion_plan and path and path_index < len(path):
                        next_position = path[path_index]
                        player.center = next_position
                        path_index += 1
                        pygame.time.wait(25)

                    
                    dx, dy = controller(player_speed)
                    new_player = player.move(dx, dy)
                    if (
                        0 <= new_player.centerx < 512 and 0 <= new_player.centery < 512 and
                        not any(pygame.Rect(point[0], point[1], 1, 1).colliderect(new_player) for ast_obstacle in ast_obstacles for point in ast_obstacle)
                    ):
                        player = new_player
                    for lava_mask in lava_masks:
                        offset = (player.x, player.y)
                        if lava_mask.overlap(pygame.mask.Mask((player.width, player.height), fill=True), offset):
                            player = initial_player_position
                            path_index = 0
                            break
                    if player.colliderect(objective_rect):
                        game_won = True

                screen.blit(background_image, (0, 0))
                pygame.draw.rect(screen, (255, 255, 255, 100), reset_button)
                reset_text = pygame.font.Font(None, 24).render("Reset", True, (0, 0, 0))
                screen.blit(reset_text, (reset_button.x + 5, reset_button.y + 5))

                if visualize and path:
                    for i in range(len(path) - 1):
                        pygame.draw.line(screen, (255, 255, 0), path[i], path[i + 1], 2)

                screen.blit(free_space_surface, (0, 0))
                
                # Draw the lava and asteroids
                for simplex in lava_triangles:
                    triangle = [free_space_points[i] for i in simplex]
                    lava_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
                    pygame.draw.polygon(lava_surface, (255, 255, 255), triangle)
                    scaled_texture = pygame.transform.scale(lava_texture, lava_surface.get_size())
                    lava_surface.blit(scaled_texture, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                    screen.blit(lava_surface, (0, 0))

                for ast_obstacle in ast_obstacles:
                    ast_obstacle_surface = pygame.Surface((512, 512), pygame.SRCALPHA)
                    pygame.draw.polygon(ast_obstacle_surface, (255, 255, 255), ast_obstacle)
                    scaled_texture = pygame.transform.scale(asteroid_texture, ast_obstacle_surface.get_size())
                    ast_obstacle_surface.blit(scaled_texture, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
                    screen.blit(ast_obstacle_surface, (0, 0))

                screen.blit(bright_rocket, (player.x - 8.5, player.y))
                screen.blit(objective_texture, (objective_rect.x, objective_rect.y))

                if game_won:
                    font = pygame.font.Font(None, 74)
                    text = font.render("Player Won!", True, (255, 255, 255))
                    screen.blit(text, (150, 200))
                    restart_button = pygame.Rect(200, 300, 120, 50)
                    pygame.draw.rect(screen, (0, 255, 0), restart_button)
                    restart_text = pygame.font.Font(None, 36).render("Restart", True, (0, 0, 0))
                    screen.blit(restart_text, (restart_button.x + 15, restart_button.y + 10))

                if visualize:
                    pygame.draw.rect(screen, (0, 255, 0), player, 2)
                    pygame.draw.rect(screen, (0, 0, 255), objective_rect, 2)
                    for simplex in lava_triangles:
                        triangle = [free_space_points[i] for i in simplex]
                        pygame.draw.polygon(screen, (255, 0, 0), triangle, 2)
                    triangulation_surface = pygame.Surface((512, 512), pygame.SRCALPHA)

                    for simplex, color in zip(triangulation.simplices, triangle_colors):
                        triangle = [free_space_points[i] for i in simplex]
                        pygame.draw.polygon(triangulation_surface, color, triangle)

                    screen.blit(triangulation_surface, (0, 0))

                    for expanded_obstacle in expanded_obstacles:
                        pygame.draw.polygon(free_space_surface, (0, 255, 0, 50), expanded_obstacle)

                    for expanded_lava in expanded_lava_triangles:
                        pygame.draw.polygon(free_space_surface, (255, 0, 0, 50), expanded_lava)

                pygame.display.flip()
                clock.tick(30)
            pygame.quit()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()