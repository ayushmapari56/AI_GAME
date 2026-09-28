import pygame
import math
from typing import List, Tuple, Dict, Optional, Set
from algorithms.astar import astar_search
from algorithms.unsupervised import ZoneClusterer
from algorithms.supervised import PlayerPredictor

class HunterAI:
    """
    Multi-Tier AI Enemy Hunter featuring:
      - Level 1: Pure A* Pathfinding (Direct Chase)
      - Level 2: Unsupervised Zone-Aware Tracking (K-Means Cost Modulation)
      - Level 3: Supervised Predictive Interception (Trajectory Anticipation)
    """
    def __init__(self, start_pos: Tuple[int, int], cell_size: int = 32):
        self.grid_x = start_pos[0]
        self.grid_y = start_pos[1]
        self.cell_size = cell_size

        # Render coordinates for smooth lerping
        self.render_x = float(start_pos[0] * cell_size)
        self.render_y = float(start_pos[1] * cell_size)

        # AI Configuration & Level
        self.ai_level = 1 # 1: A*, 2: K-Means, 3: Supervised Interceptor
        self.base_speed = 0.20 # seconds per grid step (lower = faster)
        self.step_timer = 0.0
        self.stun_timer = 0.0

        # Pathfinding & Telemetry Cache
        self.current_path: List[Tuple[int, int]] = []
        self.explored_nodes: List[Tuple[int, int]] = []
        self.open_set: Set[Tuple[int, int]] = set()
        self.closed_set: Set[Tuple[int, int]] = set()
        self.target_cell: Tuple[int, int] = start_pos
        self.predicted_cell: Tuple[int, int] = start_pos
        self.prediction_confidence: float = 0.0
        self.predicted_dir_str: str = ""
        self.last_search_cost: float = 0.0
        self.last_execution_time_ms: float = 0.0
        self.nodes_explored_count: int = 0
        self.is_intercepting: bool = False

        # Visuals
        self.pulse_phase = 0.0

    def reset(self, start_pos: Tuple[int, int], ai_level: int = 1):
        self.grid_x, self.grid_y = start_pos
        self.render_x = float(start_pos[0] * self.cell_size)
        self.render_y = float(start_pos[1] * self.cell_size)
        self.ai_level = ai_level
        self.step_timer = 0.0
        self.stun_timer = 0.0
        self.current_path.clear()
        self.explored_nodes.clear()
        self.open_set.clear()
        self.closed_set.clear()
        self.target_cell = start_pos
        self.predicted_cell = start_pos
        self.prediction_confidence = 0.0
        self.predicted_dir_str = ""

    def stun(self, duration: float = 3.0):
        self.stun_timer = max(self.stun_timer, duration)

    def compute_path(
        self,
        grid: List[List[int]],
        player_pos: Tuple[int, int],
        decoy_pos: Optional[Tuple[int, int]],
        clusterer: Optional[ZoneClusterer],
        predictor: Optional[PlayerPredictor]
    ):
        """Calculates path using the algorithm tier configured for this level."""
        enemy_pos = (self.grid_x, self.grid_y)

        # 1. Determine Target Cell
        if decoy_pos is not None:
            # Player decoy distracts hunter
            self.target_cell = decoy_pos
            self.is_intercepting = False
        elif self.ai_level == 1:
            # Level 1: Direct Player Chase
            self.target_cell = player_pos
            self.is_intercepting = False
        elif self.ai_level == 2:
            # Level 2: Direct Chase with Zone-Aware Traversal
            self.target_cell = player_pos
            self.is_intercepting = False
        elif self.ai_level >= 3:
            # Level 3: Supervised Learning Predictive Interception
            if predictor is not None:
                pred_pos, pred_dir, conf, _ = predictor.predict_next_target(
                    player_pos, enemy_pos, grid, lookahead_steps=2
                )
                self.predicted_cell = pred_pos
                self.predicted_dir_str = pred_dir
                self.prediction_confidence = conf

                # Intercept predicted spot if distance is sufficient
                manh_dist = abs(enemy_pos[0] - player_pos[0]) + abs(enemy_pos[1] - player_pos[1])
                if manh_dist > 2 and conf >= 0.40:
                    self.target_cell = pred_pos
                    self.is_intercepting = True
                else:
                    self.target_cell = player_pos
                    self.is_intercepting = False
            else:
                self.target_cell = player_pos
                self.is_intercepting = False

        # 2. Zone Cost Modulation for Level 2 and 3
        cost_map = None
        if self.ai_level >= 2 and clusterer is not None and clusterer.is_trained:
            cost_map = clusterer.get_zone_cost_map()

        # 3. Execute A* Search
        result = astar_search(
            start=enemy_pos,
            goal=self.target_cell,
            grid=grid,
            cost_map=cost_map,
            heuristic="manhattan"
        )

        self.current_path = result["path"]
        self.explored_nodes = result["explored_nodes"]
        self.open_set = result["open_set"]
        self.closed_set = result["closed_set"]
        self.last_search_cost = result["cost"]
        self.last_execution_time_ms = result["execution_time_ms"]
        self.nodes_explored_count = result["nodes_visited_count"]

    def update(
        self,
        dt: float,
        grid: List[List[int]],
        player_pos: Tuple[int, int],
        decoy_pos: Optional[Tuple[int, int]],
        clusterer: Optional[ZoneClusterer],
        predictor: Optional[PlayerPredictor]
    ) -> bool:
        """
        Updates enemy AI step and smooth lerp.
        Returns True if caught the player on this step.
        """
        self.pulse_phase += dt * 4.0

        if self.stun_timer > 0:
            self.stun_timer -= dt
            # Render position lerps towards current grid cell
            self.render_x += (self.grid_x * self.cell_size - self.render_x) * min(1.0, dt * 10.0)
            self.render_y += (self.grid_y * self.cell_size - self.render_y) * min(1.0, dt * 10.0)
            return False

        # Calculate current step speed (faster in hot zones in Level 2/3)
        current_speed = self.base_speed
        if self.ai_level >= 2 and clusterer is not None:
            zid, _, _ = clusterer.get_zone_info((self.grid_x, self.grid_y))
            if zid == 2: # Hotspot
                current_speed *= 0.75 # 25% faster hunt speed
            elif zid == 0:
                current_speed *= 1.10

        self.step_timer += dt
        if self.step_timer >= current_speed:
            self.step_timer = 0.0

            # Recompute optimal route
            self.compute_path(grid, player_pos, decoy_pos, clusterer, predictor)

            # Take step along path if available
            if len(self.current_path) > 1:
                next_step = self.current_path[1]
                self.grid_x, self.grid_y = next_step
                self.current_path.pop(0)

        # Smooth render position lerp
        target_rx = float(self.grid_x * self.cell_size)
        target_ry = float(self.grid_y * self.cell_size)
        lerp_rate = 14.0
        self.render_x += (target_rx - self.render_x) * min(1.0, dt * lerp_rate)
        self.render_y += (target_ry - self.render_y) * min(1.0, dt * lerp_rate)

        # Check collision with player
        return (self.grid_x, self.grid_y) == player_pos

    def draw(
        self,
        surface: pygame.Surface,
        offset_x: int = 0,
        offset_y: int = 0,
        show_path: bool = True,
        show_explored: bool = False
    ):
        # 1. Visualize Explored Nodes if toggled
        if show_explored and self.explored_nodes:
            for cell in self.explored_nodes:
                cx = cell[0] * self.cell_size + offset_x
                cy = cell[1] * self.cell_size + offset_y
                exp_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                exp_surf.fill((255, 100, 50, 45))
                surface.blit(exp_surf, (cx, cy))

        # 2. Draw A* Planned Path Line / Glow
        if show_path and len(self.current_path) > 1:
            points = [
                (c[0] * self.cell_size + self.cell_size // 2 + offset_x,
                 c[1] * self.cell_size + self.cell_size // 2 + offset_y)
                for c in self.current_path
            ]
            
            # Laser glow line
            path_color = (255, 60, 60, 180) if not self.is_intercepting else (255, 170, 0, 220)
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                pygame.draw.line(surface, path_color[:3], p1, p2, 3)

            # Draw dots on route
            for pt in points[1:]:
                pygame.draw.circle(surface, path_color[:3], pt, 3)

        # 3. Draw Predictive Target Marker if intercepting in Level 3
        if self.ai_level >= 3 and self.is_intercepting:
            tx = self.predicted_cell[0] * self.cell_size + offset_x
            ty = self.predicted_cell[1] * self.cell_size + offset_y
            
            target_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            # Pulsing target reticle
            pulse_rad = int(self.cell_size * 0.35 + math.sin(self.pulse_phase) * 3)
            pygame.draw.circle(target_surf, (255, 165, 0, 160), (self.cell_size // 2, self.cell_size // 2), pulse_rad, 2)
            pygame.draw.circle(target_surf, (255, 50, 50, 200), (self.cell_size // 2, self.cell_size // 2), 4)
            surface.blit(target_surf, (tx, ty))

        # 4. Draw Devil / Demon Hunter Character
        ex = int(self.render_x + offset_x)
        ey = int(self.render_y + offset_y)
        cx = ex + self.cell_size // 2
        cy = ey + self.cell_size // 2
        radius = int(self.cell_size * 0.44)

        # Devil Aura Flame Glow
        pulse_alpha = int(60 + 35 * math.sin(self.pulse_phase))
        threat_color = (220, 38, 38, pulse_alpha) if self.ai_level < 3 else (234, 88, 12, pulse_alpha)
        glow_surf = pygame.Surface((self.cell_size * 2, self.cell_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, threat_color, (self.cell_size, self.cell_size), int(radius * 1.35))
        surface.blit(glow_surf, (cx - self.cell_size, cy - self.cell_size))

        # Demon Wings / Cloak Spikes
        wing_color = (127, 29, 29)
        # Left wing
        pygame.draw.polygon(surface, wing_color, [
            (cx - int(radius * 0.3), cy),
            (cx - int(radius * 1.1), cy - int(radius * 0.4)),
            (cx - int(radius * 0.9), cy + int(radius * 0.5)),
            (cx - int(radius * 0.2), cy + int(radius * 0.3))
        ])
        # Right wing
        pygame.draw.polygon(surface, wing_color, [
            (cx + int(radius * 0.3), cy),
            (cx + int(radius * 1.1), cy - int(radius * 0.4)),
            (cx + int(radius * 0.9), cy + int(radius * 0.5)),
            (cx + int(radius * 0.2), cy + int(radius * 0.3))
        ])

        # Sharp Devil Horns
        horn_color = (15, 23, 42)
        # Left Horn
        pygame.draw.polygon(surface, horn_color, [
            (cx - int(radius * 0.4), cy - int(radius * 0.1)),
            (cx - int(radius * 0.7), cy - int(radius * 0.9)),
            (cx - int(radius * 0.15), cy - int(radius * 0.35))
        ])
        # Right Horn
        pygame.draw.polygon(surface, horn_color, [
            (cx + int(radius * 0.4), cy - int(radius * 0.1)),
            (cx + int(radius * 0.7), cy - int(radius * 0.9)),
            (cx + int(radius * 0.15), cy - int(radius * 0.35))
        ])

        # Demon Head & Fierce Jaw
        demon_face_color = (220, 38, 38) if self.ai_level < 3 else (234, 88, 12)
        face_pts = [
            (cx, cy - int(radius * 0.5)),
            (cx + int(radius * 0.6), cy - int(radius * 0.1)),
            (cx + int(radius * 0.45), cy + int(radius * 0.45)),
            (cx, cy + int(radius * 0.85)), # Pointy Chin
            (cx - int(radius * 0.45), cy + int(radius * 0.45)),
            (cx - int(radius * 0.6), cy - int(radius * 0.1))
        ]
        pygame.draw.polygon(surface, demon_face_color, face_pts)
        pygame.draw.polygon(surface, (254, 202, 202), face_pts, 1)

        # Piercing Demon Eyes
        eye_color = (251, 191, 36) if self.stun_timer <= 0 else (56, 189, 248)
        # Left eye
        pygame.draw.polygon(surface, eye_color, [
            (cx - int(radius * 0.4), cy - int(radius * 0.1)),
            (cx - int(radius * 0.15), cy - int(radius * 0.05)),
            (cx - int(radius * 0.25), cy + int(radius * 0.05))
        ])
        # Right eye
        pygame.draw.polygon(surface, eye_color, [
            (cx + int(radius * 0.4), cy - int(radius * 0.1)),
            (cx + int(radius * 0.15), cy - int(radius * 0.05)),
            (cx + int(radius * 0.25), cy + int(radius * 0.05))
        ])

        # Evil Fangs
        pygame.draw.polygon(surface, (255, 255, 255), [
            (cx - int(radius * 0.18), cy + int(radius * 0.3)),
            (cx - int(radius * 0.10), cy + int(radius * 0.48)),
            (cx - int(radius * 0.02), cy + int(radius * 0.3))
        ])
        pygame.draw.polygon(surface, (255, 255, 255), [
            (cx + int(radius * 0.18), cy + int(radius * 0.3)),
            (cx + int(radius * 0.10), cy + int(radius * 0.48)),
            (cx + int(radius * 0.02), cy + int(radius * 0.3))
        ])

        # Stun effect
        if self.stun_timer > 0:
            stun_surf = pygame.Surface((self.cell_size * 2, self.cell_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(stun_surf, (56, 189, 248, 160), (self.cell_size, self.cell_size), int(radius * 1.4), 2)
            surface.blit(stun_surf, (cx - self.cell_size, cy - self.cell_size))
