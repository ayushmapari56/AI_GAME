import pygame
from typing import Tuple, List, Optional, Dict
from collections import deque

class Player:
    def __init__(self, start_pos: Tuple[int, int], cell_size: int = 32):
        self.grid_x = start_pos[0]
        self.grid_y = start_pos[1]
        self.target_grid_x = start_pos[0]
        self.target_grid_y = start_pos[1]
        self.cell_size = cell_size

        # Render coordinates for sub-cell lerping
        self.render_x = float(start_pos[0] * cell_size)
        self.render_y = float(start_pos[1] * cell_size)

        # Movement state
        self.speed = 9.0 # lerp rate
        self.move_cooldown = 0.12 # seconds between steps
        self.cooldown_timer = 0.0
        self.queued_dir: Optional[Tuple[int, int]] = None
        self.last_move_dir: Tuple[int, int] = (0, 0)

        # Game metrics
        self.score = 0
        self.orbs_collected = 0
        self.steps_taken = 0
        self.history: deque = deque(maxlen=200) # (x, y, timestamp)
        self.trail_particles: List[Dict] = []

        # Buffs & Special powers
        self.speed_boost_timer = 0.0
        self.decoy_active = False
        self.decoy_pos: Optional[Tuple[int, int]] = None
        self.decoy_timer = 0.0
        self.emp_charges = 1
        self.is_alive = True

    def reset(self, start_pos: Tuple[int, int]):
        self.grid_x, self.grid_y = start_pos
        self.target_grid_x, self.target_grid_y = start_pos
        self.render_x = float(start_pos[0] * self.cell_size)
        self.render_y = float(start_pos[1] * self.cell_size)
        self.queued_dir = None
        self.last_move_dir = (0, 0)
        self.cooldown_timer = 0.0
        self.trail_particles.clear()
        self.speed_boost_timer = 0.0
        self.decoy_active = False
        self.decoy_pos = None
        self.decoy_timer = 0.0
        self.is_alive = True

    def queue_input(self, dx: int, dy: int):
        self.queued_dir = (dx, dy)

    def update(self, dt: float, grid: List[List[int]]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Updates player position and timers.
        Returns (prev_pos, new_pos) if a step occurred, else None.
        """
        step_result = None

        # Update timers
        if self.cooldown_timer > 0:
            self.cooldown_timer -= dt
        if self.speed_boost_timer > 0:
            self.speed_boost_timer -= dt
        if self.decoy_timer > 0:
            self.decoy_timer -= dt
            if self.decoy_timer <= 0:
                self.decoy_active = False
                self.decoy_pos = None

        current_cooldown = self.move_cooldown * 0.6 if self.speed_boost_timer > 0 else self.move_cooldown

        # Process grid movement when ready
        if self.cooldown_timer <= 0 and self.queued_dir is not None:
            dx, dy = self.queued_dir
            nx = self.grid_x + dx
            ny = self.grid_y + dy

            rows = len(grid)
            cols = len(grid[0]) if rows > 0 else 0

            if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
                prev = (self.grid_x, self.grid_y)
                self.grid_x = nx
                self.grid_y = ny
                self.last_move_dir = (dx, dy)
                self.steps_taken += 1
                self.history.append((nx, ny))
                self.cooldown_timer = current_cooldown
                step_result = (prev, (nx, ny))

                # Add trail particle
                self.trail_particles.append({
                    "x": prev[0] * self.cell_size + self.cell_size / 2,
                    "y": prev[1] * self.cell_size + self.cell_size / 2,
                    "life": 1.0
                })

        # Smooth render position lerping
        target_rx = float(self.grid_x * self.cell_size)
        target_ry = float(self.grid_y * self.cell_size)
        lerp_speed = 18.0 if self.speed_boost_timer > 0 else 14.0
        self.render_x += (target_rx - self.render_x) * min(1.0, dt * lerp_speed)
        self.render_y += (target_ry - self.render_y) * min(1.0, dt * lerp_speed)

        # Update particle trails
        for p in self.trail_particles:
            p["life"] -= dt * 2.5
        self.trail_particles = [p for p in self.trail_particles if p["life"] > 0]

        return step_result

    def deploy_decoy(self):
        if not self.decoy_active:
            self.decoy_active = True
            self.decoy_pos = (self.grid_x, self.grid_y)
            self.decoy_timer = 4.0 # lasts 4 seconds
            return True
        return False

    def trigger_emp(self) -> bool:
        if self.emp_charges > 0:
            self.emp_charges -= 1
            return True
        return False

    def draw(self, surface: pygame.Surface, offset_x: int = 0, offset_y: int = 0):
        # Draw particle trails
        for p in self.trail_particles:
            alpha = int(p["life"] * 160)
            radius = int(self.cell_size * 0.25 * p["life"])
            if radius > 0:
                s = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (0, 240, 255, alpha), (radius, radius), radius)
                surface.blit(s, (p["x"] - radius + offset_x, p["y"] - radius + offset_y))

        # Draw Decoy Hologram if active
        if self.decoy_active and self.decoy_pos:
            dx_px = self.decoy_pos[0] * self.cell_size + offset_x
            dy_px = self.decoy_pos[1] * self.cell_size + offset_y
            decoy_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.circle(
                decoy_surf,
                (0, 200, 255, 120),
                (self.cell_size // 2, self.cell_size // 2),
                int(self.cell_size * 0.38)
            )
            pygame.draw.circle(
                decoy_surf,
                (255, 255, 255, 180),
                (self.cell_size // 2, self.cell_size // 2),
                int(self.cell_size * 0.45),
                2
            )
            surface.blit(decoy_surf, (dx_px, dy_px))

        # Draw Human Player Character
        px = int(self.render_x + offset_x)
        py = int(self.render_y + offset_y)
        cx = px + self.cell_size // 2
        cy = py + self.cell_size // 2
        rad = int(self.cell_size * 0.40)

        # Soft shadow
        shadow_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 50), (int(self.cell_size * 0.2), int(self.cell_size * 0.75), int(self.cell_size * 0.6), int(self.cell_size * 0.2)))
        surface.blit(shadow_surf, (px, py))

        # Human Torso & Cyber Vest
        vest_color = (37, 99, 235) if self.speed_boost_timer <= 0 else (217, 119, 6)
        vest_rect = pygame.Rect(cx - int(rad * 0.6), cy - int(rad * 0.2), int(rad * 1.2), int(rad * 0.8))
        pygame.draw.rect(surface, vest_color, vest_rect, border_radius=4)

        # Chest Core
        core_color = (255, 255, 255) if self.speed_boost_timer <= 0 else (254, 240, 138)
        pygame.draw.circle(surface, core_color, (cx, cy + int(rad * 0.15)), int(rad * 0.2))

        # Human Head
        pygame.draw.circle(surface, (253, 224, 71), (cx, cy - int(rad * 0.42)), int(rad * 0.38))
        # Hair/Cap
        pygame.draw.circle(surface, (15, 23, 42), (cx, cy - int(rad * 0.55)), int(rad * 0.38))

        # Visor
        dx, dy = self.last_move_dir
        look_x = dx * int(rad * 0.25)
        look_y = dy * int(rad * 0.2)
        visor_rect = pygame.Rect(cx - int(rad * 0.3) + look_x, cy - int(rad * 0.45) + look_y, int(rad * 0.6), int(rad * 0.22))
        pygame.draw.rect(surface, (56, 189, 248), visor_rect, border_radius=2)
