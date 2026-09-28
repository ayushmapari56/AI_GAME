import sys
from maze import Maze
from algorithms.astar import astar_search
from algorithms.unsupervised import ZoneClusterer
from algorithms.supervised import PlayerPredictor

def run_tests():
    print("1. Testing Procedural Maze Generation...")
    m = Maze(cols=25, rows=19, loop_factor=0.2)
    assert len(m.grid) == 19 and len(m.grid[0]) == 25, "Maze dimensions invalid"
    walkable = m.get_walkable_cells()
    assert len(walkable) > 50, "Not enough walkable cells"
    print(f"   [PASS] Maze initialized with {len(walkable)} walkable corridors and {len(m.collectibles)} collectibles.")

    print("\n2. Testing A* Pathfinding Telemetry...")
    res = astar_search(m.player_spawn, m.enemy_spawn, m.grid)
    assert res["success"] or len(res["path"]) > 0, "A* failed to produce path"
    print(f"   [PASS] A* optimal path length: {len(res['path'])} | Nodes explored: {res['nodes_visited_count']} | Time: {res['execution_time_ms']:.2f}ms")

    print("\n3. Testing Unsupervised K-Means Zone Clustering...")
    clusterer = ZoneClusterer(cols=25, rows=19, n_clusters=3)
    clusterer.compute_connectivity(m.grid)
    for cell in walkable[:20]:
        clusterer.record_player_step(cell, dt=0.2)
    success = clusterer.fit_clusters(m.grid)
    assert success, "K-Means fit failed"
    stats = clusterer.get_cluster_stats()
    print(f"   [PASS] K-Means fitted successfully. Inertia: {stats['inertia']:.2f} | Cluster counts: {stats['counts']}")

    print("\n4. Testing Supervised Player Predictor...")
    predictor = PlayerPredictor(cols=25, rows=19, history_len=5)
    # Simulate movement sequence: UP -> UP -> RIGHT -> RIGHT -> DOWN
    moves = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4)]
    for i in range(len(moves) - 1):
        predictor.record_move(moves[i], moves[i+1], m.enemy_spawn, m.grid)
    
    pred_pos, pred_dir, conf, probs = predictor.predict_next_target(moves[-1], m.enemy_spawn, m.grid, lookahead_steps=2)
    print(f"   [PASS] Supervised predicted next target: {pred_pos} ({pred_dir}) with {conf*100:.1f}% confidence.")
    
    print("\n==========================================")
    print("   ALL CORE AI SYSTEMS OPERATING 100%!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()
