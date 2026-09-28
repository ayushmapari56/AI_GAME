import heapq
import time
from typing import List, Tuple, Dict, Set, Optional

class AStarNode:
    def __init__(self, pos: Tuple[int, int], g_cost: float = float('inf'), h_cost: float = 0.0, parent: Optional['AStarNode'] = None):
        self.pos = pos
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other: 'AStarNode'):
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost

def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def reconstruct_path(current_node: AStarNode) -> List[Tuple[int, int]]:
    path = []
    curr = current_node
    while curr:
        path.append(curr.pos)
        curr = curr.parent
    path.reverse()
    return path

def astar_search(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: List[List[int]], # 0 for passage, 1 for wall
    cost_map: Optional[Dict[Tuple[int, int], float]] = None,
    heuristic: str = "manhattan"
) -> Dict:
    """
    Executes A* search and returns comprehensive telemetry for visualization.
    Returns:
        {
            "path": List[Tuple[int, int]],
            "explored_nodes": List[Tuple[int, int]], # ordered sequence of exploration
            "open_set": Set[Tuple[int, int]],
            "closed_set": Set[Tuple[int, int]],
            "cost": float,
            "nodes_visited_count": int,
            "execution_time_ms": float,
            "success": bool
        }
    """
    start_time = time.perf_counter()
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    if not (0 <= start[0] < cols and 0 <= start[1] < rows):
        return {"path": [], "explored_nodes": [], "open_set": set(), "closed_set": set(), "cost": 0, "nodes_visited_count": 0, "execution_time_ms": 0, "success": False}

    # If goal is inside a wall or out of bounds, find closest walkable cell
    if not (0 <= goal[0] < cols and 0 <= goal[1] < rows) or grid[goal[1]][goal[0]] == 1:
        # Fallback to nearest walkable neighbor
        best_dist = float('inf')
        walkable_goal = start
        for y in range(rows):
            for x in range(cols):
                if grid[y][x] == 0:
                    d = manhattan_distance((x, y), goal)
                    if d < best_dist:
                        best_dist = d
                        walkable_goal = (x, y)
        goal = walkable_goal

    h_func = manhattan_distance if heuristic == "manhattan" else euclidean_distance

    start_node = AStarNode(pos=start, g_cost=0.0, h_cost=h_func(start, goal))
    open_heap: List[AStarNode] = [start_node]
    open_dict: Dict[Tuple[int, int], AStarNode] = {start: start_node}
    closed_set: Set[Tuple[int, int]] = set()
    explored_order: List[Tuple[int, int]] = []

    best_node_so_far = start_node
    best_h = start_node.h_cost

    while open_heap:
        current = heapq.heappop(open_heap)
        
        # Check if already processed with better cost
        if current.pos in closed_set:
            continue

        closed_set.add(current.pos)
        explored_order.append(current.pos)

        if current.h_cost < best_h:
            best_h = current.h_cost
            best_node_so_far = current

        if current.pos == goal:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path = reconstruct_path(current)
            return {
                "path": path,
                "explored_nodes": explored_order,
                "open_set": set(open_dict.keys()) - closed_set,
                "closed_set": closed_set,
                "cost": current.g_cost,
                "nodes_visited_count": len(closed_set),
                "execution_time_ms": elapsed_ms,
                "success": True
            }

        # 4-directional cardinal neighbors
        x, y = current.pos
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for nx, ny in neighbors:
            if 0 <= nx < cols and 0 <= ny < rows:
                if grid[ny][nx] == 1:
                    continue # wall
                
                npos = (nx, ny)
                if npos in closed_set:
                    continue

                # Base cost is 1.0, plus optional dynamic zone cost multiplier
                step_cost = 1.0
                if cost_map and npos in cost_map:
                    step_cost = max(0.1, cost_map[npos])

                tentative_g = current.g_cost + step_cost
                
                if npos in open_dict and tentative_g >= open_dict[npos].g_cost:
                    continue

                neighbor_node = AStarNode(
                    pos=npos,
                    g_cost=tentative_g,
                    h_cost=h_func(npos, goal),
                    parent=current
                )
                open_dict[npos] = neighbor_node
                heapq.heappush(open_heap, neighbor_node)

    # If goal is unreachable, return path to closest explored cell
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    path = reconstruct_path(best_node_so_far)
    return {
        "path": path,
        "explored_nodes": explored_order,
        "open_set": set(open_dict.keys()) - closed_set,
        "closed_set": closed_set,
        "cost": best_node_so_far.g_cost,
        "nodes_visited_count": len(closed_set),
        "execution_time_ms": elapsed_ms,
        "success": False
    }
