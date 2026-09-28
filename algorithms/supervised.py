import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import List, Tuple, Dict, Optional
from collections import deque

DIR_MAP = {
    (0, -1): 0, # UP
    (0, 1): 1,  # DOWN
    (-1, 0): 2, # LEFT
    (1, 0): 3,  # RIGHT
    (0, 0): 4   # STAY
}
REV_DIR_MAP = {v: k for k, v in DIR_MAP.items()}
DIR_NAMES = {0: "UP [^]", 1: "DOWN [v]", 2: "LEFT [<]", 3: "RIGHT [>]", 4: "STAY [-]"}


class PlayerPredictor:
    """
    Supervised Learning Module using Decision Trees / Random Forests & Markov Chains.
    Predicts the player's next move direction and destination coordinates.
    """
    def __init__(self, cols: int, rows: int, history_len: int = 5):
        self.cols = cols
        self.rows = rows
        self.history_len = history_len

        # Training dataset
        self.features: List[List[float]] = []
        self.labels: List[int] = []

        # Real-time state tracking
        self.recent_positions = deque(maxlen=history_len)
        self.recent_actions = deque(maxlen=history_len)
        self.model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
        self.is_trained = False
        self.train_accuracy = 0.0
        self.total_predictions = 0
        self.correct_predictions = 0
        self.last_predicted_dir: Optional[int] = None
        self.last_confidence: float = 0.0
        self.dir_probabilities: Dict[str, float] = {}

        # 1st-order Markov Transition Matrix for instant cold-start
        self.markov_counts = np.zeros((5, 5), dtype=np.float32)

    def extract_cell_features(
        self,
        player_pos: Tuple[int, int],
        enemy_pos: Tuple[int, int],
        grid: List[List[int]],
        last_dir_code: int = 4
    ) -> List[float]:
        """Builds a normalized feature vector describing the current game state."""
        px, py = player_pos
        ex, ey = enemy_pos

        # Wall sensors around player (1 = blocked, 0 = open)
        wall_up = 1.0 if (py - 1 < 0 or grid[py - 1][px] == 1) else 0.0
        wall_down = 1.0 if (py + 1 >= self.rows or grid[py + 1][px] == 1) else 0.0
        wall_left = 1.0 if (px - 1 < 0 or grid[py][px - 1] == 1) else 0.0
        wall_right = 1.0 if (px + 1 >= self.cols or grid[py][px + 1] == 1) else 0.0

        # Relative vector to enemy
        dx_enemy = (ex - px) / max(1, self.cols)
        dy_enemy = (ey - py) / max(1, self.rows)
        dist_enemy = (abs(ex - px) + abs(ey - py)) / max(1, self.cols + self.rows)

        # Normalized coordinates
        norm_px = px / max(1, self.cols - 1)
        norm_py = py / max(1, self.rows - 1)

        # One-hot encoded last direction (5 classes)
        last_dir_onehot = [1.0 if last_dir_code == i else 0.0 for i in range(5)]

        return [
            norm_px, norm_py,
            wall_up, wall_down, wall_left, wall_right,
            dx_enemy, dy_enemy, dist_enemy,
            *last_dir_onehot
        ]

    def record_move(
        self,
        prev_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
        enemy_pos: Tuple[int, int],
        grid: List[List[int]]
    ):
        """Records a transition step for supervised training."""
        dx = new_pos[0] - prev_pos[0]
        dy = new_pos[1] - prev_pos[1]
        action_code = DIR_MAP.get((dx, dy), 4)

        last_action = self.recent_actions[-1] if len(self.recent_actions) > 0 else 4
        feat = self.extract_cell_features(prev_pos, enemy_pos, grid, last_action)

        self.features.append(feat)
        self.labels.append(action_code)

        # Update Markov counts
        self.markov_counts[last_action, action_code] += 1.0

        # Check prediction accuracy
        if self.last_predicted_dir is not None:
            self.total_predictions += 1
            if self.last_predicted_dir == action_code:
                self.correct_predictions += 1

        self.recent_positions.append(new_pos)
        self.recent_actions.append(action_code)

    def train_model(self) -> bool:
        """Trains the Random Forest on recorded player trajectories."""
        # Need at least 10 samples and more than 1 class
        if len(self.features) < 8 or len(set(self.labels)) <= 1:
            return False

        X = np.array(self.features, dtype=np.float32)
        y = np.array(self.labels, dtype=np.int32)

        try:
            self.model.fit(X, y)
            self.is_trained = True
            self.train_accuracy = float(self.model.score(X, y))
            return True
        except Exception:
            return False

    def predict_next_target(
        self,
        player_pos: Tuple[int, int],
        enemy_pos: Tuple[int, int],
        grid: List[List[int]],
        lookahead_steps: int = 2
    ) -> Tuple[Tuple[int, int], str, float, Dict[str, float]]:
        """
        Predicts the player's next move and projects lookahead coordinates.
        Returns:
            (predicted_target_pos, predicted_direction_name, confidence, probabilities_dict)
        """
        last_action = self.recent_actions[-1] if len(self.recent_actions) > 0 else 4
        feat = self.extract_cell_features(player_pos, enemy_pos, grid, last_action)
        
        prob_dict = {DIR_NAMES[i]: 0.0 for i in range(5)}
        best_dir = 4
        confidence = 0.5

        if self.is_trained:
            try:
                X_test = np.array([feat], dtype=np.float32)
                probs = self.model.predict_proba(X_test)[0]
                classes = self.model.classes_
                
                for cls_idx, cls_label in enumerate(classes):
                    name = DIR_NAMES.get(cls_label, "STAY •")
                    prob_dict[name] = float(probs[cls_idx])

                best_dir = int(classes[np.argmax(probs)])
                confidence = float(np.max(probs))
            except Exception:
                best_dir = last_action
        else:
            # Cold-start Markov transition fallback
            row = self.markov_counts[last_action]
            row_sum = np.sum(row)
            if row_sum > 0:
                markov_probs = row / row_sum
                for i in range(5):
                    prob_dict[DIR_NAMES[i]] = float(markov_probs[i])
                best_dir = int(np.argmax(markov_probs))
                confidence = float(np.max(markov_probs))
            else:
                # Default to momentum (same direction) if open
                best_dir = last_action if last_action in REV_DIR_MAP else 4
                prob_dict[DIR_NAMES[best_dir]] = 0.6
                confidence = 0.6

        # Filter out impossible moves (walking straight into a wall)
        px, py = player_pos
        step_dx, step_dy = REV_DIR_MAP.get(best_dir, (0, 0))
        target_x, target_y = px + step_dx, py + step_dy

        if not (0 <= target_x < self.cols and 0 <= target_y < self.rows) or grid[target_y][target_x] == 1:
            # Find the best valid direction according to probabilities
            valid_dirs = []
            for d_code, (dx, dy) in REV_DIR_MAP.items():
                nx, ny = px + dx, py + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows and grid[ny][nx] == 0:
                    valid_dirs.append(d_code)
            
            if valid_dirs:
                # Pick valid direction with highest probability or stay
                valid_dirs.sort(key=lambda d: prob_dict.get(DIR_NAMES[d], 0.0), reverse=True)
                best_dir = valid_dirs[0]
                step_dx, step_dy = REV_DIR_MAP[best_dir]
            else:
                best_dir = 4
                step_dx, step_dy = (0, 0)

        self.last_predicted_dir = best_dir
        self.last_confidence = confidence
        self.dir_probabilities = prob_dict

        # Project lookahead positions
        curr_x, curr_y = px, py
        for _ in range(max(1, lookahead_steps)):
            nx = curr_x + step_dx
            ny = curr_y + step_dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows and grid[ny][nx] == 0:
                curr_x, curr_y = nx, ny
            else:
                break

        return (curr_x, curr_y), DIR_NAMES[best_dir], confidence, prob_dict

    def get_stats(self) -> Dict:
        accuracy = (self.correct_predictions / max(1, self.total_predictions)) * 100.0
        return {
            "samples_count": len(self.features),
            "is_trained": self.is_trained,
            "train_accuracy": self.train_accuracy * 100.0,
            "live_accuracy": accuracy,
            "total_predictions": self.total_predictions,
            "last_confidence": self.last_confidence * 100.0,
            "last_predicted_dir": DIR_NAMES.get(self.last_predicted_dir, "N/A") if self.last_predicted_dir is not None else "N/A"
        }
