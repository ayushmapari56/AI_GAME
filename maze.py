import random
from typing import List, Tuple, Set, Dict

class Maze:
    """
    Procedural Maze Generator with configurable loops, corridors, collectibles, and power-ups.
    """
    def __init__(self, cols: int = 25, rows: int = 19, loop_factor: float = 0.15):
        # Ensure odd dimensions for proper wall/corridor grid
        self.cols = cols if cols % 2 == 1 else cols + 1
        self.rows = rows if rows % 2 == 1 else rows + 1
        self.loop_factor = loop_factor
        
        self.grid: List[List[int]] = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
        self.collectibles: Set[Tuple[int, int]] = set()
        self.powerups: Dict[Tuple[int, int], str] = {} # (x, y) -> 'speed', 'decoy', 'freeze'
        self.exit_portal: Tuple[int, int] = (self.cols - 2, 1)
        self.player_spawn: Tuple[int, int] = (1, 1)
        self.enemy_spawn: Tuple[int, int] = (self.cols - 2, self.rows - 2)

        self.generate()

    def generate(self, seed: int = None):
        if seed is not None:
            random.seed(seed)

        # 1 = Wall, 0 = Walkable passage
        self.grid = [[1 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Recursive backtracker from (1, 1)
        stack = [(1, 1)]
        self.grid[1][1] = 0
        visited = {(1, 1)}

        while stack:
            cx, cy = stack[-1]
            neighbors = []
            
            # Step by 2 in cardinal directions
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 1 <= nx < self.cols - 1 and 1 <= ny < self.rows - 1:
                    if (nx, ny) not in visited:
                        neighbors.append((nx, ny, cx + dx // 2, cy + dy // 2))

            if neighbors:
                nx, ny, wx, wy = random.choice(neighbors)
                self.grid[wy][wx] = 0
                self.grid[ny][nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()

        # Add loops to allow tactical flanking and multiple paths
        extra_loops = int((self.cols * self.rows) * self.loop_factor * 0.1)
        for _ in range(extra_loops):
            rx = random.randrange(1, self.cols - 1)
            ry = random.randrange(1, self.rows - 1)
            if self.grid[ry][rx] == 1:
                # Carve wall if connecting two passages
                horizontal = (self.grid[ry][rx - 1] == 0 and self.grid[ry][rx + 1] == 0)
                vertical = (self.grid[ry - 1][rx] == 0 and self.grid[ry + 1][rx] == 0)
                if horizontal or vertical:
                    self.grid[ry][rx] = 0

        # Spawns & Exits
        self.player_spawn = (1, 1)
        self.enemy_spawn = (self.cols - 2, self.rows - 2)
        self.exit_portal = (self.cols - 2, 1)

        self.grid[self.player_spawn[1]][self.player_spawn[0]] = 0
        self.grid[self.enemy_spawn[1]][self.enemy_spawn[0]] = 0
        self.grid[self.exit_portal[1]][self.exit_portal[0]] = 0

        # Spawn Collectibles (Data Nodes / Energy Orbs)
        self.collectibles.clear()
        self.powerups.clear()

        walkable = self.get_walkable_cells()
        random.shuffle(walkable)

        # Place collectibles on ~35% of walkable cells
        num_collectibles = max(10, int(len(walkable) * 0.35))
        for cell in walkable[:num_collectibles]:
            if cell != self.player_spawn and cell != self.enemy_spawn and cell != self.exit_portal:
                self.collectibles.add(cell)

        # Place 2-3 tactical powerups
        powerup_types = ['speed', 'decoy', 'emp']
        remaining = [c for c in walkable if c not in self.collectibles and c != self.player_spawn and c != self.enemy_spawn]
        random.shuffle(remaining)
        for i, cell in enumerate(remaining[:3]):
            self.powerups[cell] = powerup_types[i % len(powerup_types)]

    def get_walkable_cells(self) -> List[Tuple[int, int]]:
        cells = []
        for y in range(self.rows):
            for x in range(self.cols):
                if self.grid[y][x] == 0:
                    cells.append((x, y))
        return cells

    def is_walkable(self, x: int, y: int) -> bool:
        if 0 <= x < self.cols and 0 <= y < self.rows:
            return self.grid[y][x] == 0
        return False
