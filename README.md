# 🎮 AI Maze Hunter: Multi-Tier Machine Learning & Search Game

**AI Maze Hunter** is an interactive, real-time AI demonstration game built in **Python + Pygame + Scikit-Learn**. It showcases how three fundamentally distinct AI paradigms collaborate to create an intelligent, adaptive opponent that learns, predicts, and navigates.

---

## 🧠 The 3 AI Pillars & Their Roles

```
                 PLAYER
                   ↓
          Player movement data
                   ↓
        ┌──────────────────────┐
        │ Supervised Learning  │  --> Predicts where the player is heading (Random Forest + Markov)
        │ Predict next target  │
        └──────────┬───────────┘
                   ↓
             Predicted target
                   ↓
        ┌──────────────────────┐
        │     A* SEARCH        │  --> Calculates optimal path & modulates cost via clusters
        │  Find optimal route  │
        └──────────┬───────────┘
                   ↓
             AI ENEMY
                   ↑
                   │
        ┌──────────────────────┐
        │ Unsupervised Learning│  --> Dynamically segments maze into Safe, Transit & Danger zones
        │ Cluster map zones    │
        └──────────────────────┘
```

| AI Technique | Model / Method | Role in Game |
| :--- | :--- | :--- |
| **Supervised Learning** | `RandomForestClassifier` + Markov Lookahead | Predicts the player's next move direction and projects 1–3 steps into the future. |
| **Unsupervised Learning** | `KMeans(k=3)` with spatio-temporal telemetry | Clusters the maze grid into 🟢 Safe (Zone 0), 🟡 Transit (Zone 1), and 🔴 Hotspot (Zone 2) regions. |
| **A* Graph Search** | Heuristic Graph Traversal ($f(n) = g(n) + h(n)$) | Computes the shortest path from the AI Hunter to the player's predicted interception coordinate. |

---

## 🕹️ 3 Game Levels

### **Level 1 — A* Maze (The Classical Hunter)**
- Pure Graph Search pathfinding.
- Visualizes explored nodes, open set, closed set, and execution speed in real-time.
- Direct player pursuit.

### **Level 2 — Unsupervised World (Zone-Aware AI)**
- K-Means continuously clusters map tiles based on player visit counts, dwell duration, and proximity encounters.
- 🟢 **Green Zone**: Safe / Low Activity
- 🟡 **Yellow Zone**: Moderate Activity / Corridors
- 🔴 **Red Zone**: Danger Hotspot / Ambush Corridor
- AI Hunter gains speed and traverses hot zones with priority edge weights.

### **Level 3 — Learning Hunter (Predictive Interceptor)**
- Live online supervised model training on player movement history.
- Displays live probability distributions for `[UP, DOWN, LEFT, RIGHT]`.
- AI Hunter targets the **predicted future cell** rather than where you are now, anticipating your route to cut you off!

---

## 🎯 Controls & Hotkeys

| Key | Action |
| :--- | :--- |
| **Arrow Keys / WASD** | Move Player |
| **SPACE** | Deploy Decoy Hologram (diverts AI pursuit) |
| **E** | Fire EMP Shockwave (stuns hunter for 3.5s) |
| **V** | Toggle A* Explored Nodes overlay |
| **H** | Toggle K-Means Zone Heatmap overlay |
| **P** | Toggle Predictive Interception Ghost Reticle |
| **1 / 2 / 3** | Instant Switch AI Level |
| **R** | Restart Level / Generate New Procedural Maze |
| **TAB** | Open Full Deep ML Architecture Guide & Formulas |
| **ESC** | Exit Game / Close Modal |

---

## 🚀 Installation & Running

```bash
# 1. Install dependencies
pip install pygame scikit-learn numpy scipy

# 2. Run the game
python main.py
```
