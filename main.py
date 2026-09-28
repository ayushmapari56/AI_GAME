import pygame
import sys
import time
import math
from typing import Dict, Tuple, Optional

from maze import Maze
from player import Player
from enemy import HunterAI
from sound import SoundManager
from ui import UI
from algorithms.unsupervised import ZoneClusterer
from algorithms.supervised import PlayerPredictor

# Game Window Constants
MAZE_COLS = 25
MAZE_ROWS = 19
CELL_SIZE = 34
SIDEBAR_WIDTH = 340

SCREEN_WIDTH = MAZE_COLS * CELL_SIZE + SIDEBAR_WIDTH
SCREEN_HEIGHT = MAZE_ROWS * CELL_SIZE

# Game State Enums
STATE_PLAYING = 1
STATE_LEVEL_CLEAR = 2
STATE_GAME_OVER = 3
STATE_ALL_CLEARED = 4

class AIMazeHunterGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("AI Maze Hunter — Supervised + Unsupervised + A* Search")
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True

        # Modules
        self.sound = SoundManager()
        self.ui = UI(SCREEN_WIDTH, SCREEN_HEIGHT, SIDEBAR_WIDTH)

        # Game State
        self.current_level = 1
        self.game_state = STATE_PLAYING
        self.show_ml_modal = False
        
        # Visual Toggles
        self.toggles = {
            "explored": True,
            "heatmap": True,
            "ghost": True
        }

        # Timers
        self.kmeans_train_timer = 0.0
        self.supervised_train_timer = 0.0
        self.animation_time = 0.0

        self.init_level(self.current_level)

    def init_level(self, level: int):
        self.current_level = level
        self.game_state = STATE_PLAYING

        # Generate Maze
        self.maze = Maze(cols=MAZE_COLS, rows=MAZE_ROWS, loop_factor=0.18 + level * 0.05)
        self.total_orbs = len(self.maze.collectibles)

        # Entities
        self.player = Player(start_pos=self.maze.player_spawn, cell_size=CELL_SIZE)
        self.enemy = HunterAI(start_pos=self.maze.enemy_spawn, cell_size=CELL_SIZE)
        self.enemy.ai_level = level

        # ML Engines
        self.clusterer = ZoneClusterer(cols=MAZE_COLS, rows=MAZE_ROWS, n_clusters=3)
        self.clusterer.compute_connectivity(self.maze.grid)
        
        self.predictor = PlayerPredictor(cols=MAZE_COLS, rows=MAZE_ROWS, history_len=5)

        # Initial enemy path compute
        self.enemy.compute_path(
            self.maze.grid,
            (self.player.grid_x, self.player.grid_y),
            self.player.decoy_pos,
            self.clusterer if level >= 2 else None,
            self.predictor if level >= 3 else None
        )

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                # Toggle Deep ML modal
                if event.key == pygame.K_TAB:
                    self.show_ml_modal = not self.show_ml_modal
                elif event.key == pygame.K_ESCAPE:
                    if self.show_ml_modal:
                        self.show_ml_modal = False
                    else:
                        self.running = False

                # Level Switch Hotkeys
                elif event.key == pygame.K_1:
                    self.init_level(1)
                elif event.key == pygame.K_2:
                    self.init_level(2)
                elif event.key == pygame.K_3:
                    self.init_level(3)

                # Visual Toggles
                elif event.key == pygame.K_v:
                    self.toggles["explored"] = not self.toggles["explored"]
                elif event.key == pygame.K_h:
                    self.toggles["heatmap"] = not self.toggles["heatmap"]
                elif event.key == pygame.K_p:
                    self.toggles["ghost"] = not self.toggles["ghost"]

                # Game Actions
                elif event.key == pygame.K_SPACE:
                    if self.game_state == STATE_PLAYING:
                        if self.player.deploy_decoy():
                            self.sound.play("powerup")
                    elif self.game_state in (STATE_LEVEL_CLEAR, STATE_GAME_OVER, STATE_ALL_CLEARED):
                        self.advance_or_restart()

                elif event.key == pygame.K_e:
                    if self.game_state == STATE_PLAYING and self.player.trigger_emp():
                        self.enemy.stun(duration=3.5)
                        self.sound.play("emp")

                elif event.key == pygame.K_r:
                    self.init_level(self.current_level)

                elif event.key == pygame.K_RETURN:
                    if self.game_state in (STATE_LEVEL_CLEAR, STATE_GAME_OVER, STATE_ALL_CLEARED):
                        self.advance_or_restart()

    def advance_or_restart(self):
        if self.game_state == STATE_LEVEL_CLEAR:
            if self.current_level < 3:
                self.init_level(self.current_level + 1)
            else:
                self.game_state = STATE_ALL_CLEARED
        elif self.game_state in (STATE_GAME_OVER, STATE_ALL_CLEARED):
            self.init_level(1)

    def process_input(self):
        if self.game_state != STATE_PLAYING or self.show_ml_modal:
            return

        keys = pygame.key.get_pressed()
        dx, dy = 0, 0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = 1
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -1
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = 1

        if dx != 0 or dy != 0:
            self.player.queue_input(dx, dy)

    def update(self, dt: float):
        self.animation_time += dt

        if self.game_state != STATE_PLAYING or self.show_ml_modal:
            return

        # 1. Update Player
        step = self.player.update(dt, self.maze.grid)
        player_pos = (self.player.grid_x, self.player.grid_y)
        enemy_pos = (self.enemy.grid_x, self.enemy.grid_y)

        if step is not None:
            prev_pos, new_pos = step
            
            # Feed step data into Unsupervised & Supervised modules
            self.clusterer.record_player_step(new_pos, dt)
            self.clusterer.record_enemy_proximity(new_pos, enemy_pos)
            self.predictor.record_move(prev_pos, new_pos, enemy_pos, self.maze.grid)

            # Check Collectibles
            if new_pos in self.maze.collectibles:
                self.maze.collectibles.remove(new_pos)
                self.player.score += 50
                self.player.orbs_collected += 1
                self.sound.play("collect")

            # Check Power-ups
            if new_pos in self.maze.powerups:
                ptype = self.maze.powerups.pop(new_pos)
                if ptype == 'speed':
                    self.player.speed_boost_timer = 4.0
                elif ptype == 'emp':
                    self.player.emp_charges += 1
                elif ptype == 'decoy':
                    self.player.deploy_decoy()
                self.sound.play("powerup")

            # Check Exit Portal
            if new_pos == self.maze.exit_portal and len(self.maze.collectibles) == 0:
                self.sound.play("victory")
                self.game_state = STATE_LEVEL_CLEAR

        # 2. Periodic Unsupervised Clustering Update (every 2.0 seconds in Level >= 2)
        if self.current_level >= 2:
            self.kmeans_train_timer += dt
            if self.kmeans_train_timer >= 2.0:
                self.kmeans_train_timer = 0.0
                self.clusterer.fit_clusters(self.maze.grid)

        # 3. Periodic Supervised Predictor Retraining (every 1.5 seconds in Level >= 3)
        if self.current_level >= 3:
            self.supervised_train_timer += dt
            if self.supervised_train_timer >= 1.5:
                self.supervised_train_timer = 0.0
                self.predictor.train_model()

        # 4. Update Enemy Hunter AI
        caught = self.enemy.update(
            dt=dt,
            grid=self.maze.grid,
            player_pos=player_pos,
            decoy_pos=self.player.decoy_pos,
            clusterer=self.clusterer if self.current_level >= 2 else None,
            predictor=self.predictor if self.current_level >= 3 else None
        )

        if caught:
            self.sound.play("gameover")
            self.game_state = STATE_GAME_OVER

    def draw_maze(self):
        # 1. Draw Maze Floor
        for y in range(MAZE_ROWS):
            for x in range(MAZE_COLS):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if self.maze.grid[y][x] == 1:
                    # Wall Tile with High-Tech styling
                    pygame.draw.rect(self.screen, (20, 26, 36), rect)
                    pygame.draw.rect(self.screen, (36, 46, 62), rect, 1)
                else:
                    # Floor Passage
                    pygame.draw.rect(self.screen, (10, 13, 18), rect)

                    # Draw K-Means Zone Heatmap if enabled (Level >= 2)
                    if self.current_level >= 2 and self.toggles["heatmap"] and self.clusterer.is_trained:
                        zid, _, col = self.clusterer.get_zone_info((x, y))
                        alpha = 50 if zid == 0 else (75 if zid == 1 else 100)
                        zone_surf = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                        zone_surf.fill((*col, alpha))
                        self.screen.blit(zone_surf, (x * CELL_SIZE, y * CELL_SIZE))

        # 2. Draw Collectibles (Energy Dots)
        for cx, cy in self.maze.collectibles:
            center = (cx * CELL_SIZE + CELL_SIZE // 2, cy * CELL_SIZE + CELL_SIZE // 2)
            pulse = math.sin(self.animation_time * 5.0 + cx * 0.5 + cy * 0.5) * 1.5
            rad = max(3, int(4 + pulse))
            pygame.draw.circle(self.screen, (255, 230, 80), center, rad)
            pygame.draw.circle(self.screen, (255, 255, 255), center, max(1, rad - 2))

        # 3. Draw Tactical Power-ups
        for (px, py), ptype in self.maze.powerups.items():
            center = (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2)
            if ptype == 'speed':
                pygame.draw.circle(self.screen, (0, 255, 200), center, int(CELL_SIZE * 0.32), 2)
                pygame.draw.circle(self.screen, (0, 255, 200), center, 4)
            elif ptype == 'emp':
                pygame.draw.circle(self.screen, (200, 100, 255), center, int(CELL_SIZE * 0.32), 2)
                pygame.draw.circle(self.screen, (200, 100, 255), center, 4)
            elif ptype == 'decoy':
                pygame.draw.circle(self.screen, (100, 200, 255), center, int(CELL_SIZE * 0.32), 2)
                pygame.draw.circle(self.screen, (100, 200, 255), center, 4)

        # 4. Draw Exit Portal
        ex, ey = self.maze.exit_portal
        portal_center = (ex * CELL_SIZE + CELL_SIZE // 2, ey * CELL_SIZE + CELL_SIZE // 2)
        portal_active = (len(self.maze.collectibles) == 0)
        p_col = (0, 255, 180) if portal_active else (80, 90, 105)
        
        # Rotating portal vortex
        portal_surf = pygame.Surface((CELL_SIZE * 2, CELL_SIZE * 2), pygame.SRCALPHA)
        portal_rad = int(CELL_SIZE * 0.42)
        pygame.draw.circle(portal_surf, (*p_col, 80), (CELL_SIZE, CELL_SIZE), portal_rad)
        pygame.draw.circle(portal_surf, p_col, (CELL_SIZE, CELL_SIZE), portal_rad, 2)
        self.screen.blit(portal_surf, (portal_center[0] - CELL_SIZE, portal_center[1] - CELL_SIZE))

    def render(self):
        self.screen.fill((13, 17, 23))

        # 1. Draw Maze & Entities
        self.draw_maze()
        
        # Enemy with Path & Explored Nodes
        self.enemy.draw(
            self.screen,
            offset_x=0,
            offset_y=0,
            show_path=True,
            show_explored=(self.toggles["explored"] and self.current_level >= 1)
        )
        
        # Player
        self.player.draw(self.screen, offset_x=0, offset_y=0)

        # 2. Draw Live Telemetry Sidebar
        astar_telemetry = {
            "nodes_visited_count": self.enemy.nodes_explored_count,
            "execution_time_ms": self.enemy.last_execution_time_ms,
            "cost": self.enemy.last_search_cost,
            "path": self.enemy.current_path
        }
        cluster_telemetry = self.clusterer.get_cluster_stats()
        predictor_telemetry = self.predictor.get_stats()
        prob_dict = self.predictor.dir_probabilities

        self.ui.draw_sidebar(
            surface=self.screen,
            level=self.current_level,
            score=self.player.score,
            orbs_left=len(self.maze.collectibles),
            total_orbs=self.total_orbs,
            astar_data=astar_telemetry,
            cluster_data=cluster_telemetry,
            predictor_data=predictor_telemetry,
            prob_dict=prob_dict,
            toggles=self.toggles,
            player_emp=self.player.emp_charges,
            player_decoy=self.player.decoy_active
        )

        # 3. Draw Game Over / Level Clear Modals
        if self.game_state == STATE_LEVEL_CLEAR:
            next_name = f"Proceed to Level {self.current_level + 1}" if self.current_level < 3 else "Final Victory!"
            self.ui.draw_banner_modal(
                self.screen,
                "MAZE CLEARED!",
                f"Completed Level {self.current_level} | Orbs Gathered: {self.player.orbs_collected}",
                (46, 204, 113),
                f"Press [ENTER] to {next_name}"
            )
        elif self.game_state == STATE_GAME_OVER:
            self.ui.draw_banner_modal(
                self.screen,
                "CAUGHT BY AI HUNTER!",
                f"The AI intercepted you using {['A* Search', 'K-Means Zone Strategy', 'Supervised Prediction Interception'][self.current_level - 1]}",
                (239, 68, 68),
                "Press [ENTER] to Try Again"
            )
        elif self.game_state == STATE_ALL_CLEARED:
            self.ui.draw_banner_modal(
                self.screen,
                "🏆 AI MASTER ARCHITECT!",
                f"You outsmarted all 3 AI algorithms! Final Score: {self.player.score}",
                (168, 85, 247),
                "Press [ENTER] to Play Again"
            )

        # 4. Draw Full Deep ML Info Overlay if active
        if self.show_ml_modal:
            self.ui.draw_deep_ml_modal(self.screen)

        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.process_input()
            self.update(dt)
            self.render()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = AIMazeHunterGame()
    game.run()
