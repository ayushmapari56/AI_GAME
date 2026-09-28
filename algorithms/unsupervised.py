import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional

class ZoneClusterer:
    """
    Unsupervised Learning Module using K-Means Clustering.
    Analyzes spatial telemetry and player behavior to segment the maze into:
      🟢 Zone 0 (Green): Safe / Infrequently Patrolled / Low Activity
      🟡 Zone 1 (Yellow): Moderate Activity / Transit Corridor
      🔴 Zone 2 (Red): Hotspot / High Danger / Heavy Player Presence
    """
    def __init__(self, cols: int, rows: int, n_clusters: int = 3):
        self.cols = cols
        self.rows = rows
        self.n_clusters = n_clusters
        
        # Cell-level metrics
        self.visit_counts = np.zeros((rows, cols), dtype=np.float32)
        self.enemy_proximity_hits = np.zeros((rows, cols), dtype=np.float32)
        self.dwell_time = np.zeros((rows, cols), dtype=np.float32)
        self.connectivity = np.zeros((rows, cols), dtype=np.float32) # open neighbors

        # Clustering results
        self.cluster_map = np.zeros((rows, cols), dtype=np.int32) # cluster id for each cell
        self.cluster_labels_meaning: Dict[int, str] = {0: "Safe", 1: "Transit", 2: "Hotspot"}
        self.cluster_colors: Dict[int, Tuple[int, int, int]] = {
            0: (46, 204, 113),   # Emerald Green
            1: (241, 196, 15),   # Sun Yellow
            2: (231, 76, 60)     # Crimson Red
        }
        self.cluster_costs: Dict[int, float] = {0: 1.0, 1: 1.2, 2: 0.8} # Enemy prefers hot zones to hunt
        self.is_trained = False
        self.inertia = 0.0
        self.centroids = None
        self.update_counter = 0

    def compute_connectivity(self, grid: List[List[int]]):
        """Precomputes open passage degree for every cell."""
        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y][x] == 0:
                    open_count = 0
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.cols and 0 <= ny < self.rows and grid[ny][nx] == 0:
                            open_count += 1
                    self.connectivity[y, x] = open_count

    def record_player_step(self, pos: Tuple[int, int], dt: float = 0.1):
        x, y = pos
        if 0 <= x < self.cols and 0 <= y < self.rows:
            self.visit_counts[y, x] += 1.0
            self.dwell_time[y, x] += dt
            self.update_counter += 1

    def record_enemy_proximity(self, player_pos: Tuple[int, int], enemy_pos: Tuple[int, int]):
        dist = abs(player_pos[0] - enemy_pos[0]) + abs(player_pos[1] - enemy_pos[1])
        if dist <= 4:
            px, py = player_pos
            if 0 <= px < self.cols and 0 <= py < self.rows:
                self.enemy_proximity_hits[py, px] += max(0.0, 5.0 - dist)

    def fit_clusters(self, grid: List[List[int]]) -> bool:
        """
        Runs K-Means clustering on the feature matrix of walkable cells.
        Features per cell: [Normalized Visits, Normalized Dwell, Proximity Hits, Connectivity, Normalized X, Normalized Y]
        """
        walkable_coords = []
        features = []

        max_visits = max(1.0, float(np.max(self.visit_counts)))
        max_dwell = max(1.0, float(np.max(self.dwell_time)))
        max_hits = max(1.0, float(np.max(self.enemy_proximity_hits)))

        for y in range(self.rows):
            for x in range(self.cols):
                if grid[y][x] == 0:
                    v_norm = self.visit_counts[y, x] / max_visits
                    d_norm = self.dwell_time[y, x] / max_dwell
                    h_norm = self.enemy_proximity_hits[y, x] / max_hits
                    c_norm = self.connectivity[y, x] / 4.0
                    x_norm = x / max(1, self.cols - 1)
                    y_norm = y / max(1, self.rows - 1)

                    # Weights: Player activity is most dominant
                    feat = [
                        v_norm * 3.0,
                        d_norm * 2.5,
                        h_norm * 2.0,
                        c_norm * 1.0,
                        x_norm * 0.5,
                        y_norm * 0.5
                    ]
                    walkable_coords.append((x, y))
                    features.append(feat)

        if len(features) < self.n_clusters:
            return False

        X = np.array(features, dtype=np.float32)
        try:
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=3, max_iter=100, random_state=42)
            raw_labels = kmeans.fit_predict(X)
            self.inertia = float(kmeans.inertia_)
            self.centroids = kmeans.cluster_centers_

            # Sort clusters by average activity level so:
            # Cluster 0 = Lowest Activity (Safe / Green)
            # Cluster 1 = Moderate Activity (Transit / Yellow)
            # Cluster 2 = Highest Activity (Hotspot / Red)
            cluster_activity_scores = []
            for k in range(self.n_clusters):
                mask = (raw_labels == k)
                if np.any(mask):
                    # Activity score based on visit and dwell features (first 2 features)
                    activity_score = np.mean(X[mask][:, 0]) + np.mean(X[mask][:, 1]) + np.mean(X[mask][:, 2])
                else:
                    activity_score = float(k)
                cluster_activity_scores.append((activity_score, k))

            # Map raw cluster ID -> sorted rank (0, 1, 2)
            cluster_activity_scores.sort(key=lambda item: item[0])
            label_mapping = {old_k: new_rank for new_rank, (_, old_k) in enumerate(cluster_activity_scores)}

            # Populate grid
            self.cluster_map.fill(0)
            for (x, y), raw_lbl in zip(walkable_coords, raw_labels):
                sorted_lbl = label_mapping[raw_lbl]
                self.cluster_map[y, x] = sorted_lbl

            self.is_trained = True
            return True
        except Exception:
            return False

    def get_zone_cost_map(self) -> Dict[Tuple[int, int], float]:
        """Returns cell coordinates -> movement cost for A* search."""
        cost_map = {}
        for y in range(self.rows):
            for x in range(self.cols):
                zone_id = int(self.cluster_map[y, x])
                # Hotspots are faster/cheaper for AI hunter to traverse, Safe zones cost standard
                cost_map[(x, y)] = self.cluster_costs.get(zone_id, 1.0)
        return cost_map

    def get_zone_info(self, pos: Tuple[int, int]) -> Tuple[int, str, Tuple[int, int, int]]:
        x, y = pos
        if 0 <= x < self.cols and 0 <= y < self.rows:
            zid = int(self.cluster_map[y, x])
            return zid, self.cluster_labels_meaning.get(zid, "Normal"), self.cluster_colors.get(zid, (100, 100, 100))
        return 0, "Unknown", (100, 100, 100)

    def get_cluster_stats(self) -> Dict:
        """Returns breakdown of cells per zone."""
        total_walkable = max(1, int(np.sum(self.cluster_map >= 0)))
        counts = [int(np.sum(self.cluster_map == k)) for k in range(self.n_clusters)]
        percentages = [(c / total_walkable) * 100.0 for c in counts]
        return {
            "counts": counts,
            "percentages": percentages,
            "inertia": self.inertia,
            "is_trained": self.is_trained,
            "total_updates": self.update_counter
        }
