/**
 * AI MAZE HUNTER — CORE GAME & TRI-ALGORITHM ENGINE
 * Implements:
 *   1. A* Graph Search with Exploratory Telemetry & Edge Modulation
 *   2. Unsupervised K-Means Clustering for Spatial Zone Segmentation
 *   3. Supervised Online Classifier for Player Trajectory Prediction
 */

// --- Priority Queue for A* Search ---
class PriorityQueue {
  constructor() {
    this.elements = [];
  }
  enqueue(item, priority) {
    this.elements.push({ item, priority });
    this.elements.sort((a, b) => a.priority - b.priority);
  }
  dequeue() {
    return this.elements.shift().item;
  }
  isEmpty() {
    return this.elements.length === 0;
  }
}

// --- Procedural Synth Audio Generator (Web Audio API) ---
class WebAudioSynth {
  constructor() {
    this.ctx = null;
    this.enabled = true;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
  }

  playTone(freqStart, freqEnd, type, duration, vol = 0.15) {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }

    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = type;
    const now = this.ctx.currentTime;
    osc.frequency.setValueAtTime(freqStart, now);
    osc.frequency.exponentialRampToValueAtTime(Math.max(10, freqEnd), now + duration);

    gain.gain.setValueAtTime(vol, now);
    gain.gain.exponentialRampToValueAtTime(0.001, now + duration);

    osc.connect(gain);
    gain.connect(this.ctx.destination);

    osc.start(now);
    osc.stop(now + duration);
  }

  playCollect() { this.playTone(587.33, 880.0, 'sine', 0.09, 0.15); }
  playPowerup() { this.playTone(320.0, 950.0, 'triangle', 0.22, 0.2); }
  playEmp() { this.playTone(850.0, 90.0, 'sawtooth', 0.35, 0.25); }
  playAlert() { this.playTone(440.0, 440.0, 'square', 0.08, 0.1); }
  playGameOver() { this.playTone(280.0, 65.0, 'sawtooth', 0.55, 0.25); }
  playVictory() {
    this.playTone(440, 660, 'sine', 0.15, 0.2);
    setTimeout(() => this.playTone(660, 880, 'sine', 0.25, 0.2), 150);
  }
}

// ==========================================================================
// 1. A* GRAPH SEARCH IMPLEMENTATION
// ==========================================================================
function astarSearch(start, goal, grid, cols, rows, costMap = null) {
  const t0 = performance.now();
  const startKey = `${start.x},${start.y}`;
  const goalKey = `${goal.x},${goal.y}`;

  // If goal is inside a wall, find nearest walkable cell
  let effectiveGoal = { ...goal };
  if (grid[goal.y][goal.x] === 1) {
    let minDist = Infinity;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (grid[y][x] === 0) {
          const d = Math.abs(x - goal.x) + Math.abs(y - goal.y);
          if (d < minDist) {
            minDist = d;
            effectiveGoal = { x, y };
          }
        }
      }
    }
  }

  const frontier = new PriorityQueue();
  frontier.enqueue(start, 0);

  const cameFrom = new Map();
  const costSoFar = new Map();
  const exploredOrder = [];
  const closedSet = new Set();

  cameFrom.set(startKey, null);
  costSoFar.set(startKey, 0);

  let bestNode = start;
  let bestHeuristic = Math.abs(start.x - effectiveGoal.x) + Math.abs(start.y - effectiveGoal.y);

  while (!frontier.isEmpty()) {
    const current = frontier.dequeue();
    const curKey = `${current.x},${current.y}`;

    if (closedSet.has(curKey)) continue;
    closedSet.add(curKey);
    exploredOrder.push(current);

    const h = Math.abs(current.x - effectiveGoal.x) + Math.abs(current.y - effectiveGoal.y);
    if (h < bestHeuristic) {
      bestHeuristic = h;
      bestNode = current;
    }

    if (current.x === effectiveGoal.x && current.y === effectiveGoal.y) {
      bestNode = current;
      break;
    }

    const neighbors = [
      { x: current.x + 1, y: current.y },
      { x: current.x - 1, y: current.y },
      { x: current.x, y: current.y + 1 },
      { x: current.x, y: current.y - 1 }
    ];

    for (const next of neighbors) {
      if (next.x >= 0 && next.x < cols && next.y >= 0 && next.y < rows && grid[next.y][next.x] === 0) {
        const nextKey = `${next.x},${next.y}`;
        if (closedSet.has(nextKey)) continue;

        let stepCost = 1.0;
        if (costMap && costMap.has(nextKey)) {
          stepCost = costMap.get(nextKey);
        }

        const newCost = costSoFar.get(curKey) + stepCost;
        if (!costSoFar.has(nextKey) || newCost < costSoFar.get(nextKey)) {
          costSoFar.set(nextKey, newCost);
          const priority = newCost + (Math.abs(next.x - effectiveGoal.x) + Math.abs(next.y - effectiveGoal.y));
          frontier.enqueue(next, priority);
          cameFrom.set(nextKey, current);
        }
      }
    }
  }

  // Reconstruct path
  const path = [];
  let curr = bestNode;
  while (curr) {
    path.push(curr);
    curr = cameFrom.get(`${curr.x},${curr.y}`);
  }
  path.reverse();

  const elapsedMs = performance.now() - t0;
  return {
    path,
    exploredOrder,
    nodesVisited: closedSet.size,
    cost: costSoFar.get(`${bestNode.x},${bestNode.y}`) || 0,
    timeMs: elapsedMs,
    success: bestNode.x === effectiveGoal.x && bestNode.y === effectiveGoal.y
  };
}

// ==========================================================================
// 2. UNSUPERVISED LEARNING: K-MEANS ZONE CLUSTERING
// ==========================================================================
class KMeansZoneClusterer {
  constructor(cols, rows, k = 3) {
    this.cols = cols;
    this.rows = rows;
    this.k = k;

    this.visitCounts = Array.from({ length: rows }, () => new Float32Array(cols));
    this.dwellTimes = Array.from({ length: rows }, () => new Float32Array(cols));
    this.enemyHits = Array.from({ length: rows }, () => new Float32Array(cols));
    this.connectivity = Array.from({ length: rows }, () => new Float32Array(cols));
    this.clusterMap = Array.from({ length: rows }, () => new Int8Array(cols));

    this.clusterColors = [
      { r: 52, g: 211, b: 153 },  // Green (Safe)
      { r: 251, g: 191, b: 36 },  // Yellow (Transit)
      { r: 244, g: 63, b: 94 }    // Red (Hotspot)
    ];
    this.clusterCosts = [1.1, 1.0, 0.75]; // Enemy prefers red hot corridors
    this.isTrained = false;
    this.inertia = 0.0;
    this.counts = [0, 0, 0];
    this.percentages = [33, 33, 34];
  }

  computeConnectivity(grid) {
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (grid[y][x] === 0) {
          let openCount = 0;
          for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
            const nx = x + dx, ny = y + dy;
            if (nx >= 0 && nx < this.cols && ny >= 0 && ny < this.rows && grid[ny][nx] === 0) {
              openCount++;
            }
          }
          this.connectivity[y][x] = openCount;
        }
      }
    }
  }

  recordPlayerStep(pos, dt = 0.1) {
    if (pos.x >= 0 && pos.x < this.cols && pos.y >= 0 && pos.y < this.rows) {
      this.visitCounts[pos.y][pos.x] += 1.0;
      this.dwellTimes[pos.y][pos.x] += dt;
    }
  }

  recordEnemyProximity(playerPos, enemyPos) {
    const dist = Math.abs(playerPos.x - enemyPos.x) + Math.abs(playerPos.y - enemyPos.y);
    if (dist <= 4 && playerPos.y < this.rows && playerPos.x < this.cols) {
      this.enemyHits[playerPos.y][playerPos.x] += Math.max(0, 5 - dist);
    }
  }

  fit(grid) {
    const walkable = [];
    const features = [];

    let maxVisits = 1.0, maxDwell = 1.0, maxHits = 1.0;
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (grid[y][x] === 0) {
          maxVisits = Math.max(maxVisits, this.visitCounts[y][x]);
          maxDwell = Math.max(maxDwell, this.dwellTimes[y][x]);
          maxHits = Math.max(maxHits, this.enemyHits[y][x]);
        }
      }
    }

    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (grid[y][x] === 0) {
          const v = (this.visitCounts[y][x] / maxVisits) * 3.0;
          const d = (this.dwellTimes[y][x] / maxDwell) * 2.0;
          const h = (this.enemyHits[y][x] / maxHits) * 2.0;
          const c = (this.connectivity[y][x] / 4.0) * 1.0;
          const px = (x / this.cols) * 0.5;
          const py = (y / this.rows) * 0.5;

          walkable.push({ x, y });
          features.push([v, d, h, c, px, py]);
        }
      }
    }

    if (features.length < this.k) return;

    // K-Means implementation
    const numDims = features[0].length;
    let centroids = [];
    for (let i = 0; i < this.k; i++) {
      const idx = Math.floor((i / this.k) * features.length);
      centroids.push([...features[idx]]);
    }

    let assignments = new Array(features.length).fill(0);
    for (let iter = 0; iter < 12; iter++) {
      // 1. Assign points to nearest centroid
      for (let i = 0; i < features.length; i++) {
        const feat = features[i];
        let minDist = Infinity;
        let bestK = 0;
        for (let k = 0; k < this.k; k++) {
          let d = 0;
          for (let dim = 0; dim < numDims; dim++) {
            const diff = feat[dim] - centroids[k][dim];
            d += diff * diff;
          }
          if (d < minDist) {
            minDist = d;
            bestK = k;
          }
        }
        assignments[i] = bestK;
      }

      // 2. Recompute centroids
      const sums = Array.from({ length: this.k }, () => new Float32Array(numDims));
      const kCounts = new Array(this.k).fill(0);

      for (let i = 0; i < features.length; i++) {
        const k = assignments[i];
        kCounts[k]++;
        for (let dim = 0; dim < numDims; dim++) {
          sums[k][dim] += features[i][dim];
        }
      }

      for (let k = 0; k < this.k; k++) {
        if (kCounts[k] > 0) {
          for (let dim = 0; dim < numDims; dim++) {
            centroids[k][dim] = sums[k][dim] / kCounts[k];
          }
        }
      }
    }

    // Sort clusters by activity level (features 0, 1, 2)
    const clusterScores = centroids.map((c, idx) => ({
      idx,
      score: c[0] + c[1] + c[2]
    }));
    clusterScores.sort((a, b) => a.score - b.score);
    const labelMap = {};
    clusterScores.forEach((item, rank) => {
      labelMap[item.idx] = rank; // 0 = Safe, 1 = Transit, 2 = Hotspot
    });

    // Compute inertia & populate map
    let totalInertia = 0;
    this.counts = [0, 0, 0];

    for (let i = 0; i < features.length; i++) {
      const sortedK = labelMap[assignments[i]];
      const pos = walkable[i];
      this.clusterMap[pos.y][pos.x] = sortedK;
      this.counts[sortedK]++;

      // Inertia calculation
      const c = centroids[assignments[i]];
      for (let dim = 0; dim < numDims; dim++) {
        const diff = features[i][dim] - c[dim];
        totalInertia += diff * diff;
      }
    }

    const totalWalkable = Math.max(1, walkable.length);
    this.percentages = this.counts.map(c => ((c / totalWalkable) * 100));
    this.inertia = totalInertia;
    this.isTrained = true;
  }

  getCostMap() {
    const costMap = new Map();
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const zone = this.clusterMap[y][x];
        costMap.set(`${x},${y}`, this.clusterCosts[zone] || 1.0);
      }
    }
    return costMap;
  }
}

// ==========================================================================
// 3. SUPERVISED LEARNING: PLAYER TRAJECTORY PREDICTOR
// ==========================================================================
class SupervisedPredictor {
  constructor(cols, rows) {
    this.cols = cols;
    this.rows = rows;
    this.history = [];
    this.markovCounts = [
      [0, 0, 0, 0, 0], // UP
      [0, 0, 0, 0, 0], // DOWN
      [0, 0, 0, 0, 0], // LEFT
      [0, 0, 0, 0, 0], // RIGHT
      [0, 0, 0, 0, 0]  // STAY
    ];
    this.dirKeys = ['UP [^]', 'DOWN [v]', 'LEFT [<]', 'RIGHT [>]', 'STAY [-]'];
    this.dirDeltas = [
      { dx: 0, dy: -1 }, // UP
      { dx: 0, dy: 1 },  // DOWN
      { dx: -1, dy: 0 }, // LEFT
      { dx: 1, dy: 0 },  // RIGHT
      { dx: 0, dy: 0 }   // STAY
    ];
    this.lastAction = 4;
    this.lastPredictedDir = 0;
    this.lastConfidence = 0.5;
    this.totalPredictions = 0;
    this.correctPredictions = 0;
    this.probabilities = { 'UP ↑': 0.25, 'DOWN ↓': 0.25, 'LEFT ←': 0.25, 'RIGHT →': 0.25 };
  }

  recordMove(prevPos, newPos, enemyPos, grid) {
    const dx = newPos.x - prevPos.x;
    const dy = newPos.y - prevPos.y;

    let action = 4;
    if (dy === -1) action = 0;
    else if (dy === 1) action = 1;
    else if (dx === -1) action = 2;
    else if (dx === 1) action = 3;

    // Check accuracy of last prediction
    if (this.totalPredictions > 0) {
      if (this.lastPredictedDir === action) {
        this.correctPredictions++;
      }
    }
    this.totalPredictions++;

    // Update Markov model
    this.markovCounts[this.lastAction][action] += 1;
    this.lastAction = action;

    this.history.push({
      prev: prevPos,
      next: newPos,
      action
    });
  }

  predict(playerPos, enemyPos, grid, lookaheadSteps = 2) {
    const row = this.markovCounts[this.lastAction];
    const rowSum = row.reduce((a, b) => a + b, 0);

    const probs = [0.2, 0.2, 0.2, 0.2, 0.2];
    if (rowSum > 0) {
      for (let i = 0; i < 5; i++) {
        probs[i] = row[i] / rowSum;
      }
    } else {
      // Momentum bias
      probs[this.lastAction] = 0.6;
    }

    // Filter impossible moves (walls)
    let bestDir = 0;
    let maxP = -1;
    const validProbs = [0, 0, 0, 0];

    for (let d = 0; d < 4; d++) {
      const delta = this.dirDeltas[d];
      const nx = playerPos.x + delta.dx;
      const ny = playerPos.y + delta.dy;

      let score = probs[d];
      if (nx >= 0 && nx < this.cols && ny >= 0 && ny < this.rows && grid[ny][nx] === 0) {
        validProbs[d] = score + 0.1;
      } else {
        validProbs[d] = 0.001; // Wall penalty
      }

      if (validProbs[d] > maxP) {
        maxP = validProbs[d];
        bestDir = d;
      }
    }

    const normSum = validProbs.reduce((a, b) => a + b, 0) || 1;
    this.probabilities = {
      'UP ↑': validProbs[0] / normSum,
      'DOWN ↓': validProbs[1] / normSum,
      'LEFT ←': validProbs[2] / normSum,
      'RIGHT →': validProbs[3] / normSum
    };

    this.lastPredictedDir = bestDir;
    this.lastConfidence = validProbs[bestDir] / normSum;

    // Project lookahead target cell
    let target = { ...playerPos };
    const step = this.dirDeltas[bestDir];
    for (let i = 0; i < lookaheadSteps; i++) {
      const nx = target.x + step.dx;
      const ny = target.y + step.dy;
      if (nx >= 0 && nx < this.cols && ny >= 0 && ny < this.rows && grid[ny][nx] === 0) {
        target = { x: nx, y: ny };
      } else {
        break;
      }
    }

    return {
      target,
      direction: this.dirKeys[bestDir],
      confidence: this.lastConfidence,
      probabilities: this.probabilities
    };
  }
}

// ==========================================================================
// 4. PROCEDURAL MAZE GENERATOR
// ==========================================================================
class ProceduralMaze {
  constructor(cols = 25, rows = 19, loopFactor = 0.18) {
    this.cols = cols % 2 === 1 ? cols : cols + 1;
    this.rows = rows % 2 === 1 ? rows : rows + 1;
    this.loopFactor = loopFactor;
    this.grid = Array.from({ length: this.rows }, () => new Array(this.cols).fill(1));
    this.collectibles = new Set();
    this.powerups = new Map();
    this.playerSpawn = { x: 1, y: 1 };
    this.enemySpawn = { x: this.cols - 2, y: this.rows - 2 };
    this.exitPortal = { x: this.cols - 2, y: 1 };

    this.generate();
  }

  generate() {
    this.grid = Array.from({ length: this.rows }, () => new Array(this.cols).fill(1));
    const stack = [{ x: 1, y: 1 }];
    this.grid[1][1] = 0;
    const visited = new Set(['1,1']);

    while (stack.length > 0) {
      const curr = stack[stack.length - 1];
      const neighbors = [];

      for (const [dx, dy] of [[0, -2], [0, 2], [-2, 0], [2, 0]]) {
        const nx = curr.x + dx;
        const ny = curr.y + dy;
        if (nx > 0 && nx < this.cols - 1 && ny > 0 && ny < this.rows - 1) {
          if (!visited.has(`${nx},${ny}`)) {
            neighbors.push({ nx, ny, wx: curr.x + dx / 2, wy: curr.y + dy / 2 });
          }
        }
      }

      if (neighbors.length > 0) {
        const chosen = neighbors[Math.floor(Math.random() * neighbors.length)];
        this.grid[chosen.wy][chosen.wx] = 0;
        this.grid[chosen.ny][chosen.nx] = 0;
        visited.add(`${chosen.nx},${chosen.ny}`);
        stack.push({ x: chosen.nx, y: chosen.ny });
      } else {
        stack.pop();
      }
    }

    // Add loops
    const extraLoops = Math.floor((this.cols * this.rows) * this.loopFactor * 0.08);
    for (let i = 0; i < extraLoops; i++) {
      const rx = 1 + Math.floor(Math.random() * (this.cols - 2));
      const ry = 1 + Math.floor(Math.random() * (this.rows - 2));
      if (this.grid[ry][rx] === 1) {
        const h = this.grid[ry][rx - 1] === 0 && this.grid[ry][rx + 1] === 0;
        const v = this.grid[ry - 1][rx] === 0 && this.grid[ry + 1][rx] === 0;
        if (h || v) this.grid[ry][rx] = 0;
      }
    }

    // Spawns
    this.grid[this.playerSpawn.y][this.playerSpawn.x] = 0;
    this.grid[this.enemySpawn.y][this.enemySpawn.x] = 0;
    this.grid[this.exitPortal.y][this.exitPortal.x] = 0;

    // Collectibles & Power-ups
    this.collectibles.clear();
    this.powerups.clear();

    const walkable = [];
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        if (this.grid[y][x] === 0) walkable.push({ x, y });
      }
    }

    walkable.sort(() => Math.random() - 0.5);
    const numOrbs = Math.max(12, Math.floor(walkable.length * 0.32));
    for (let i = 0; i < numOrbs; i++) {
      const c = walkable[i];
      if ((c.x !== 1 || c.y !== 1) && (c.x !== this.enemySpawn.x || c.y !== this.enemySpawn.y)) {
        this.collectibles.add(`${c.x},${c.y}`);
      }
    }

    const types = ['speed', 'emp', 'decoy'];
    const rem = walkable.slice(numOrbs).filter(c => !this.collectibles.has(`${c.x},${c.y}`));
    for (let i = 0; i < Math.min(3, rem.length); i++) {
      this.powerups.set(`${rem[i].x},${rem[i].y}`, types[i % types.length]);
    }
  }
}

// ==========================================================================
// 5. MAIN GAME CONTROLLER & RENDER ENGINE
// ==========================================================================
class AIMazeHunterApp {
  constructor() {
    this.canvas = document.getElementById('gameCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.audio = new WebAudioSynth();

    this.cols = 25;
    this.rows = 19;
    this.cellSize = 34;

    this.level = 1;
    this.score = 0;
    this.empCharges = 1;
    this.gameState = 'PLAYING'; // PLAYING, LEVEL_CLEAR, GAME_OVER, ALL_CLEARED

    this.toggles = {
      explored: true,
      heatmap: true,
      ghost: true
    };

    this.player = {
      x: 1, y: 1,
      renderX: 1 * 34, renderY: 1 * 34,
      lastDir: { dx: 0, dy: 0 },
      speedBoost: 0,
      decoyActive: false,
      decoyPos: null,
      decoyTimer: 0,
      trails: []
    };

    this.enemy = {
      x: 23, y: 17,
      renderX: 23 * 34, renderY: 17 * 34,
      path: [],
      explored: [],
      nodesCount: 0,
      computeTime: 0,
      pathCost: 0,
      targetCell: { x: 1, y: 1 },
      predictedCell: { x: 1, y: 1 },
      stunTimer: 0,
      stepTimer: 0,
      baseSpeed: 0.22,
      isIntercepting: false
    };

    this.animationTime = 0;
    this.initDOM();
    this.initLevel(1);
    this.setupEvents();
    this.loop();
  }

  initDOM() {
    this.dom = {
      score: document.getElementById('scoreVal'),
      orbs: document.getElementById('orbsVal'),
      emp: document.getElementById('empVal'),
      aiMode: document.getElementById('aiModeLabel'),
      mNodes: document.getElementById('mNodesVisited'),
      mTime: document.getElementById('mComputeTime'),
      mSteps: document.getElementById('mPathSteps'),
      mCost: document.getElementById('mPathCost'),
      inertia: document.getElementById('inertiaVal'),
      segSafe: document.getElementById('segSafe'),
      segTransit: document.getElementById('segTransit'),
      segHotspot: document.getElementById('segHotspot'),
      pctSafe: document.getElementById('pctSafe'),
      pctTransit: document.getElementById('pctTransit'),
      pctHotspot: document.getElementById('pctHotspot'),
      predDir: document.getElementById('predDirText'),
      predConf: document.getElementById('predConfText'),
      pBarUp: document.getElementById('pBarUp'),
      pBarDown: document.getElementById('pBarDown'),
      pBarLeft: document.getElementById('pBarLeft'),
      pBarRight: document.getElementById('pBarRight'),
      pPctUp: document.getElementById('pPctUp'),
      pPctDown: document.getElementById('pPctDown'),
      pPctLeft: document.getElementById('pPctLeft'),
      pPctRight: document.getElementById('pPctRight'),
      supSamples: document.getElementById('supSamples'),
      supAcc: document.getElementById('supAcc'),
      bannerOverlay: document.getElementById('bannerOverlay'),
      bannerBadge: document.getElementById('bannerBadge'),
      bannerTitle: document.getElementById('bannerTitle'),
      bannerDesc: document.getElementById('bannerDesc'),
      bannerStats: document.getElementById('bannerStats'),
      bannerBtn: document.getElementById('bannerActionBtn'),
      mlModal: document.getElementById('mlModalBackdrop')
    };
  }

  initLevel(lvl) {
    this.level = lvl;
    this.gameState = 'PLAYING';
    this.dom.bannerOverlay.classList.add('hidden');

    this.maze = new ProceduralMaze(this.cols, this.rows, 0.16 + lvl * 0.05);
    this.totalOrbs = this.maze.collectibles.size;

    this.clusterer = new KMeansZoneClusterer(this.cols, this.rows, 3);
    this.clusterer.computeConnectivity(this.maze.grid);

    this.predictor = new SupervisedPredictor(this.cols, this.rows);

    this.player.x = this.maze.playerSpawn.x;
    this.player.y = this.maze.playerSpawn.y;
    this.player.renderX = this.player.x * this.cellSize;
    this.player.renderY = this.player.y * this.cellSize;
    this.player.decoyActive = false;
    this.player.decoyPos = null;
    this.player.decoyTimer = 0;
    this.player.speedBoost = 0;
    this.player.trails = [];

    this.enemy.x = this.maze.enemySpawn.x;
    this.enemy.y = this.maze.enemySpawn.y;
    this.enemy.renderX = this.enemy.x * this.cellSize;
    this.enemy.renderY = this.enemy.y * this.cellSize;
    this.enemy.stunTimer = 0;
    this.enemy.stepTimer = 0;
    this.enemy.path = [];

    // Update Tab UI
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.level) === lvl);
    });

    const modeLabels = [
      'Level 1: Direct A* Pursuit',
      'Level 2: K-Means Zone Strategy',
      'Level 3: Predictive Interception'
    ];
    this.dom.aiMode.textContent = modeLabels[lvl - 1];

    this.updateAIPath();
  }

  updateAIPath() {
    let target = { x: this.player.x, y: this.player.y };
    this.enemy.isIntercepting = false;

    // Decoy takes priority
    if (this.player.decoyActive && this.player.decoyPos) {
      target = this.player.decoyPos;
    } else if (this.level >= 3) {
      // Level 3: Supervised Lookahead Prediction
      const pred = this.predictor.predict(
        { x: this.player.x, y: this.player.y },
        { x: this.enemy.x, y: this.enemy.y },
        this.maze.grid,
        2
      );
      this.enemy.predictedCell = pred.target;

      const dist = Math.abs(this.enemy.x - this.player.x) + Math.abs(this.enemy.y - this.player.y);
      if (dist > 2 && pred.confidence >= 0.35) {
        target = pred.target;
        this.enemy.isIntercepting = true;
      }
    }

    this.enemy.targetCell = target;

    // Level 2+ Zone Cost modulation
    const costMap = (this.level >= 2 && this.clusterer.isTrained) ? this.clusterer.getCostMap() : null;

    const result = astarSearch(
      { x: this.enemy.x, y: this.enemy.y },
      target,
      this.maze.grid,
      this.cols,
      this.rows,
      costMap
    );

    this.enemy.path = result.path;
    this.enemy.explored = result.exploredOrder;
    this.enemy.nodesCount = result.nodesVisited;
    this.enemy.computeTime = result.timeMs;
    this.enemy.pathCost = result.cost;

    // Telemetry Sync
    this.dom.mNodes.textContent = result.nodesVisited;
    this.dom.mTime.textContent = `${result.timeMs.toFixed(2)} ms`;
    this.dom.mSteps.textContent = `${result.path.length} steps`;
    this.dom.mCost.textContent = result.cost.toFixed(1);
  }

  movePlayer(dx, dy) {
    if (this.gameState !== 'PLAYING') return;

    const nx = this.player.x + dx;
    const ny = this.player.y + dy;

    if (nx >= 0 && nx < this.cols && ny >= 0 && ny < this.rows && this.maze.grid[ny][nx] === 0) {
      const prev = { x: this.player.x, y: this.player.y };
      this.player.x = nx;
      this.player.y = ny;
      this.player.lastDir = { dx, dy };

      // Trail particle
      this.player.trails.push({
        x: prev.x * this.cellSize + this.cellSize / 2,
        y: prev.y * this.cellSize + this.cellSize / 2,
        life: 1.0
      });

      // Feed ML engines
      this.clusterer.recordPlayerStep({ x: nx, y: ny }, 0.2);
      this.clusterer.recordEnemyProximity({ x: nx, y: ny }, { x: this.enemy.x, y: this.enemy.y });
      this.predictor.recordMove(prev, { x: nx, y: ny }, { x: this.enemy.x, y: this.enemy.y }, this.maze.grid);

      // Collect energy orbs
      const key = `${nx},${ny}`;
      if (this.maze.collectibles.has(key)) {
        this.maze.collectibles.delete(key);
        this.score += 50;
        this.audio.playCollect();
      }

      // Collect power-ups
      if (this.maze.powerups.has(key)) {
        const ptype = this.maze.powerups.get(key);
        this.maze.powerups.delete(key);
        this.audio.playPowerup();

        if (ptype === 'speed') this.player.speedBoost = 4.0;
        else if (ptype === 'emp') this.empCharges++;
        else if (ptype === 'decoy') this.deployDecoy();
      }

      // Check Exit Portal
      if (nx === this.maze.exitPortal.x && ny === this.maze.exitPortal.y && this.maze.collectibles.size === 0) {
        this.onLevelClear();
      }

      // Retrain K-Means periodically
      if (this.level >= 2) {
        this.clusterer.fit(this.maze.grid);
      }

      this.updateAIPath();
    }
  }

  deployDecoy() {
    if (!this.player.decoyActive) {
      this.player.decoyActive = true;
      this.player.decoyPos = { x: this.player.x, y: this.player.y };
      this.player.decoyTimer = 4.0;
      this.audio.playPowerup();
      this.updateAIPath();
    }
  }

  triggerEmp() {
    if (this.empCharges > 0 && this.enemy.stunTimer <= 0) {
      this.empCharges--;
      this.enemy.stunTimer = 3.5;
      this.audio.playEmp();
    }
  }

  onLevelClear() {
    this.gameState = 'LEVEL_CLEAR';
    this.audio.playVictory();
    this.dom.bannerBadge.textContent = `LEVEL ${this.level} COMPLETED`;
    this.dom.bannerTitle.textContent = 'MAZE SECURED!';
    this.dom.bannerDesc.textContent = `You out-navigated the AI Hunter! Total Score: ${this.score}`;
    this.dom.bannerBtn.textContent = this.level < 3 ? `Advance to Level ${this.level + 1} →` : 'Play Again ↺';
    this.dom.bannerOverlay.classList.remove('hidden');
  }

  onGameOver() {
    this.gameState = 'GAME_OVER';
    this.audio.playGameOver();
    const stratNames = ['A* Shortest Path', 'K-Means Zone Interception', 'Predictive Supervised Forecasting'];
    this.dom.bannerBadge.textContent = 'CAUGHT BY HUNTER';
    this.dom.bannerTitle.textContent = 'SIMULATION FAILED';
    this.dom.bannerDesc.textContent = `The AI intercepted you using ${stratNames[this.level - 1]}.`;
    this.dom.bannerBtn.textContent = 'Try Again ↺';
    this.dom.bannerOverlay.classList.remove('hidden');
  }

  setupEvents() {
    // Keyboard Controls
    window.addEventListener('keydown', e => {
      if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') { e.preventDefault(); this.movePlayer(0, -1); }
      else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') { e.preventDefault(); this.movePlayer(0, 1); }
      else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') { e.preventDefault(); this.movePlayer(-1, 0); }
      else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') { e.preventDefault(); this.movePlayer(1, 0); }
      else if (e.code === 'Space') { e.preventDefault(); this.deployDecoy(); }
      else if (e.key === 'e' || e.key === 'E') { this.triggerEmp(); }
      else if (e.key === 'v' || e.key === 'V') {
        this.toggles.explored = !this.toggles.explored;
        document.getElementById('toggleExplored').checked = this.toggles.explored;
      }
      else if (e.key === 'h' || e.key === 'H') {
        this.toggles.heatmap = !this.toggles.heatmap;
        document.getElementById('toggleHeatmap').checked = this.toggles.heatmap;
      }
      else if (e.key === 'p' || e.key === 'P') {
        this.toggles.ghost = !this.toggles.ghost;
        document.getElementById('toggleGhost').checked = this.toggles.ghost;
      }
      else if (e.key === '1') this.initLevel(1);
      else if (e.key === '2') this.initLevel(2);
      else if (e.key === '3') this.initLevel(3);
      else if (e.key === 'r' || e.key === 'R') this.initLevel(this.level);
      else if (e.key === 'Tab') {
        e.preventDefault();
        this.dom.mlModal.classList.toggle('hidden');
      }
      else if (e.key === 'Escape') {
        this.dom.mlModal.classList.add('hidden');
      }
    });

    // Level Tabs
    document.getElementById('tabLvl1').addEventListener('click', () => this.initLevel(1));
    document.getElementById('tabLvl2').addEventListener('click', () => this.initLevel(2));
    document.getElementById('tabLvl3').addEventListener('click', () => this.initLevel(3));

    // Action Buttons
    document.getElementById('btnRestart').addEventListener('click', () => this.initLevel(this.level));
    document.getElementById('btnDecoy').addEventListener('click', () => this.deployDecoy());
    document.getElementById('btnEmp').addEventListener('click', () => this.triggerEmp());
    document.getElementById('btnAudioToggle').addEventListener('click', () => {
      this.audio.enabled = !this.audio.enabled;
      document.getElementById('audioIcon').textContent = this.audio.enabled ? '🔊' : '🔇';
    });

    // Modal
    document.getElementById('btnDocModal').addEventListener('click', () => this.dom.mlModal.classList.remove('hidden'));
    document.getElementById('btnCloseModal').addEventListener('click', () => this.dom.mlModal.classList.add('hidden'));
    document.getElementById('btnResumeGame').addEventListener('click', () => this.dom.mlModal.classList.add('hidden'));

    // Banner Modal Button
    this.dom.bannerBtn.addEventListener('click', () => {
      if (this.gameState === 'LEVEL_CLEAR') {
        this.initLevel(this.level < 3 ? this.level + 1 : 1);
      } else {
        this.initLevel(this.level);
      }
    });

    // Toggles
    document.getElementById('toggleExplored').addEventListener('change', e => this.toggles.explored = e.target.checked);
    document.getElementById('toggleHeatmap').addEventListener('change', e => this.toggles.heatmap = e.target.checked);
    document.getElementById('toggleGhost').addEventListener('change', e => this.toggles.ghost = e.target.checked);
  }

  update(dt) {
    this.animationTime += dt;

    if (this.gameState !== 'PLAYING') return;

    // Timers
    if (this.player.speedBoost > 0) this.player.speedBoost -= dt;
    if (this.player.decoyTimer > 0) {
      this.player.decoyTimer -= dt;
      if (this.player.decoyTimer <= 0) {
        this.player.decoyActive = false;
        this.player.decoyPos = null;
        this.updateAIPath();
      }
    }

    // Smooth Lerping for Player
    const pTargetX = this.player.x * this.cellSize;
    const pTargetY = this.player.y * this.cellSize;
    const pLerpSpeed = this.player.speedBoost > 0 ? 20.0 : 15.0;
    this.player.renderX += (pTargetX - this.player.renderX) * Math.min(1.0, dt * pLerpSpeed);
    this.player.renderY += (pTargetY - this.player.renderY) * Math.min(1.0, dt * pLerpSpeed);

    // Update Player Trails
    for (const trail of this.player.trails) {
      trail.life -= dt * 2.5;
    }
    this.player.trails = this.player.trails.filter(t => t.life > 0);

    // Enemy AI Step Logic
    if (this.enemy.stunTimer > 0) {
      this.enemy.stunTimer -= dt;
    } else {
      let speed = this.enemy.baseSpeed;
      if (this.level >= 2 && this.clusterer.isTrained) {
        const zid = this.clusterer.clusterMap[this.enemy.y][this.enemy.x];
        if (zid === 2) speed *= 0.75; // 25% faster in Hotspots
      }

      this.enemy.stepTimer += dt;
      if (this.enemy.stepTimer >= speed) {
        this.enemy.stepTimer = 0;
        this.updateAIPath();

        if (this.enemy.path.length > 1) {
          const next = this.enemy.path[1];
          this.enemy.x = next.x;
          this.enemy.y = next.y;
          this.enemy.path.shift();

          // Check Collision
          if (this.enemy.x === this.player.x && this.enemy.y === this.player.y) {
            this.onGameOver();
          }
        }
      }
    }

    // Smooth Lerp for Enemy
    const eTargetX = this.enemy.x * this.cellSize;
    const eTargetY = this.enemy.y * this.cellSize;
    this.enemy.renderX += (eTargetX - this.enemy.renderX) * Math.min(1.0, dt * 14.0);
    this.enemy.renderY += (eTargetY - this.enemy.renderY) * Math.min(1.0, dt * 14.0);

    // Sync Telemetry UI
    this.syncTelemetryUI();
  }

  syncTelemetryUI() {
    this.dom.score.textContent = this.score;
    this.dom.orbs.textContent = `${this.totalOrbs - this.maze.collectibles.size} / ${this.totalOrbs}`;
    this.dom.emp.textContent = this.empCharges;

    // K-Means Telemetry
    if (this.level >= 2 && this.clusterer.isTrained) {
      this.dom.inertia.textContent = `Inertia: ${this.clusterer.inertia.toFixed(1)}`;
      this.dom.segSafe.style.width = `${this.clusterer.percentages[0]}%`;
      this.dom.segTransit.style.width = `${this.clusterer.percentages[1]}%`;
      this.dom.segHotspot.style.width = `${this.clusterer.percentages[2]}%`;
      this.dom.pctSafe.textContent = `${Math.round(this.clusterer.percentages[0])}%`;
      this.dom.pctTransit.textContent = `${Math.round(this.clusterer.percentages[1])}%`;
      this.dom.pctHotspot.textContent = `${Math.round(this.clusterer.percentages[2])}%`;
    }

    // Supervised Telemetry
    if (this.level >= 3) {
      this.dom.predDir.textContent = this.predictor.dirKeys[this.predictor.lastPredictedDir];
      this.dom.predConf.textContent = `${Math.round(this.predictor.lastConfidence * 100)}%`;

      const p = this.predictor.probabilities;
      this.dom.pBarUp.style.width = `${(p['UP ↑'] || 0) * 100}%`;
      this.dom.pBarDown.style.width = `${(p['DOWN ↓'] || 0) * 100}%`;
      this.dom.pBarLeft.style.width = `${(p['LEFT ←'] || 0) * 100}%`;
      this.dom.pBarRight.style.width = `${(p['RIGHT →'] || 0) * 100}%`;

      this.dom.pPctUp.textContent = `${Math.round((p['UP ↑'] || 0) * 100)}%`;
      this.dom.pPctDown.textContent = `${Math.round((p['DOWN ↓'] || 0) * 100)}%`;
      this.dom.pPctLeft.textContent = `${Math.round((p['LEFT ←'] || 0) * 100)}%`;
      this.dom.pPctRight.textContent = `${Math.round((p['RIGHT →'] || 0) * 100)}%`;

      this.dom.supSamples.textContent = this.predictor.history.length;
      const acc = this.predictor.totalPredictions > 0
        ? ((this.predictor.correctPredictions / this.predictor.totalPredictions) * 100).toFixed(0)
        : '100';
      this.dom.supAcc.textContent = `${acc}%`;
    }
  }

  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 1. Draw Maze Tiles & Heatmap
    for (let y = 0; y < this.rows; y++) {
      for (let x = 0; x < this.cols; x++) {
        const px = x * this.cellSize;
        const py = y * this.cellSize;

        if (this.maze.grid[y][x] === 1) {
          // Architectural Slate Wall
          this.ctx.fillStyle = '#1e293b';
          this.ctx.fillRect(px, py, this.cellSize, this.cellSize);
          this.ctx.strokeStyle = '#334155';
          this.ctx.lineWidth = 1;
          this.ctx.strokeRect(px + 0.5, py + 0.5, this.cellSize - 1, this.cellSize - 1);
        } else {
          // Crisp Corridor Floor
          this.ctx.fillStyle = '#ffffff';
          this.ctx.fillRect(px, py, this.cellSize, this.cellSize);

          // Subtle Grid lines
          this.ctx.strokeStyle = '#f1f5f9';
          this.ctx.lineWidth = 1;
          this.ctx.strokeRect(px + 0.5, py + 0.5, this.cellSize - 1, this.cellSize - 1);

          // K-Means Zone Heatmap Overlay
          if (this.level >= 2 && this.toggles.heatmap && this.clusterer.isTrained) {
            const zid = this.clusterer.clusterMap[y][x];
            const col = this.clusterer.clusterColors[zid];
            const alpha = zid === 0 ? 0.15 : (zid === 1 ? 0.25 : 0.35);
            this.ctx.fillStyle = `rgba(${col.r}, ${col.g}, ${col.b}, ${alpha})`;
            this.ctx.fillRect(px, py, this.cellSize, this.cellSize);
          }
        }
      }
    }

    // 2. Draw Explored Nodes (Open/Closed Sets)
    if (this.toggles.explored && this.enemy.explored.length > 0) {
      this.ctx.fillStyle = 'rgba(37, 99, 235, 0.12)';
      for (const node of this.enemy.explored) {
        this.ctx.fillRect(node.x * this.cellSize, node.y * this.cellSize, this.cellSize, this.cellSize);
      }
    }

    // 3. Draw Collectibles (Energy Orbs)
    for (const key of this.maze.collectibles) {
      const [cx, cy] = key.split(',').map(Number);
      const px = cx * this.cellSize + this.cellSize / 2;
      const py = cy * this.cellSize + this.cellSize / 2;
      const pulse = Math.sin(this.animationTime * 6.0 + cx + cy) * 1.2;
      const radius = Math.max(3.5, 4.5 + pulse);

      this.ctx.save();
      this.ctx.fillStyle = '#d97706';
      this.ctx.beginPath();
      this.ctx.arc(px, py, radius, 0, Math.PI * 2);
      this.ctx.fill();

      this.ctx.fillStyle = '#fef08a';
      this.ctx.beginPath();
      this.ctx.arc(px - 1, py - 1, radius * 0.5, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
    }

    // 4. Draw Tactical Power-ups
    for (const [key, ptype] of this.maze.powerups.entries()) {
      const [px, py] = key.split(',').map(Number);
      const cx = px * this.cellSize + this.cellSize / 2;
      const cy = py * this.cellSize + this.cellSize / 2;

      this.ctx.save();
      this.ctx.lineWidth = 2;
      if (ptype === 'speed') {
        this.ctx.strokeStyle = '#2563eb';
        this.ctx.fillStyle = '#eff6ff';
      } else if (ptype === 'emp') {
        this.ctx.strokeStyle = '#7c3aed';
        this.ctx.fillStyle = '#f5f3ff';
      } else if (ptype === 'decoy') {
        this.ctx.strokeStyle = '#059669';
        this.ctx.fillStyle = '#ecfdf5';
      }

      this.ctx.beginPath();
      this.ctx.arc(cx, cy, this.cellSize * 0.32, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.stroke();

      // Power-up Icon Label
      this.ctx.fillStyle = ptype === 'speed' ? '#2563eb' : (ptype === 'emp' ? '#7c3aed' : '#059669');
      this.ctx.font = 'bold 11px Outfit, sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      const symbol = ptype === 'speed' ? '⚡' : (ptype === 'emp' ? 'EMP' : '🌀');
      this.ctx.fillText(symbol, cx, cy);
      this.ctx.restore();
    }

    // 5. Draw Exit Portal
    const portalX = this.maze.exitPortal.x * this.cellSize + this.cellSize / 2;
    const portalY = this.maze.exitPortal.y * this.cellSize + this.cellSize / 2;
    const portalUnlocked = this.maze.collectibles.size === 0;

    this.ctx.save();
    this.ctx.strokeStyle = portalUnlocked ? '#059669' : '#94a3b8';
    this.ctx.lineWidth = 3;
    this.ctx.beginPath();
    this.ctx.arc(portalX, portalY, this.cellSize * 0.38, 0, Math.PI * 2);
    this.ctx.stroke();

    this.ctx.fillStyle = portalUnlocked ? 'rgba(5, 150, 105, 0.15)' : 'rgba(148, 163, 184, 0.1)';
    this.ctx.fill();
    this.ctx.restore();

    // 6. Draw Decoy Hologram
    if (this.player.decoyActive && this.player.decoyPos) {
      const dx = this.player.decoyPos.x * this.cellSize + this.cellSize / 2;
      const dy = this.player.decoyPos.y * this.cellSize + this.cellSize / 2;

      this.ctx.save();
      this.ctx.strokeStyle = '#2563eb';
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([4, 4]);
      this.ctx.beginPath();
      this.ctx.arc(dx, dy, this.cellSize * 0.42, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.restore();
    }

    // 7. Draw A* Planned Path & Predictive Reticle
    if (this.enemy.path.length > 1) {
      this.ctx.save();
      this.ctx.strokeStyle = this.enemy.isIntercepting ? '#d97706' : '#dc2626';
      this.ctx.lineWidth = 3;
      this.ctx.lineCap = 'round';
      this.ctx.lineJoin = 'round';
      this.ctx.beginPath();
      for (let i = 0; i < this.enemy.path.length; i++) {
        const pt = this.enemy.path[i];
        const px = pt.x * this.cellSize + this.cellSize / 2;
        const py = pt.y * this.cellSize + this.cellSize / 2;
        if (i === 0) this.ctx.moveTo(px, py);
        else this.ctx.lineTo(px, py);
      }
      this.ctx.stroke();
      this.ctx.restore();
    }

    // Predictive Target Ghost Reticle (Level 3)
    if (this.level >= 3 && this.toggles.ghost && this.enemy.isIntercepting) {
      const gx = this.enemy.predictedCell.x * this.cellSize + this.cellSize / 2;
      const gy = this.enemy.predictedCell.y * this.cellSize + this.cellSize / 2;
      const pulse = Math.sin(this.animationTime * 8) * 2;

      this.ctx.save();
      this.ctx.strokeStyle = '#d97706';
      this.ctx.lineWidth = 2;
      this.ctx.setLineDash([3, 3]);
      this.ctx.beginPath();
      this.ctx.arc(gx, gy, this.cellSize * 0.35 + pulse, 0, Math.PI * 2);
      this.ctx.stroke();
      this.ctx.restore();
    }

    // 8. Draw Player Trails & Human Avatar
    for (const trail of this.player.trails) {
      this.ctx.save();
      this.ctx.fillStyle = `rgba(37, 99, 235, ${trail.life * 0.35})`;
      this.ctx.beginPath();
      this.ctx.arc(trail.x, trail.y, (this.cellSize * 0.20) * trail.life, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.restore();
    }

    const pRenderX = this.player.renderX + this.cellSize / 2;
    const pRenderY = this.player.renderY + this.cellSize / 2;
    this.drawHumanAvatar(
      this.ctx,
      pRenderX,
      pRenderY,
      this.cellSize * 0.40,
      this.player.lastDir,
      this.player.speedBoost,
      this.animationTime
    );

    // 9. Draw AI Devil/Demon Hunter
    const eRenderX = this.enemy.renderX + this.cellSize / 2;
    const eRenderY = this.enemy.renderY + this.cellSize / 2;
    this.drawDevilHunter(
      this.ctx,
      eRenderX,
      eRenderY,
      this.cellSize * 0.44,
      this.level,
      this.enemy.stunTimer > 0,
      this.animationTime,
      this.enemy.isIntercepting
    );
  }

  // Draw Detailed Human Character Avatar
  drawHumanAvatar(ctx, cx, cy, radius, lastDir, speedBoost, animTime) {
    ctx.save();
    ctx.translate(cx, cy);

    const isBoosted = speedBoost > 0;
    const bodyColor = isBoosted ? '#d97706' : '#2563eb';
    const skinColor = '#fde047';
    const visorColor = isBoosted ? '#fef08a' : '#38bdf8';

    // 1. Soft Ground Shadow
    ctx.fillStyle = 'rgba(15, 23, 42, 0.18)';
    ctx.beginPath();
    ctx.ellipse(0, radius * 0.85, radius * 0.65, radius * 0.25, 0, 0, Math.PI * 2);
    ctx.fill();

    // 2. Animated Running Legs
    const bob = Math.sin(animTime * 14) * 2;
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.roundRect(-radius * 0.42, radius * 0.30 + bob, radius * 0.32, radius * 0.55, 3);
    ctx.roundRect(radius * 0.10, radius * 0.30 - bob, radius * 0.32, radius * 0.55, 3);
    ctx.fill();

    // 3. Human Torso & Cyber Vest
    ctx.fillStyle = bodyColor;
    ctx.beginPath();
    ctx.roundRect(-radius * 0.62, -radius * 0.22, radius * 1.24, radius * 0.75, 5);
    ctx.fill();

    // Chest Core
    ctx.fillStyle = isBoosted ? '#fef08a' : '#ffffff';
    ctx.beginPath();
    ctx.arc(0, radius * 0.12, radius * 0.20, 0, Math.PI * 2);
    ctx.fill();

    // 4. Arms & Pumping Hands
    ctx.fillStyle = bodyColor;
    ctx.beginPath();
    ctx.arc(-radius * 0.72, radius * 0.12 - bob * 0.8, radius * 0.20, 0, Math.PI * 2);
    ctx.arc(radius * 0.72, radius * 0.12 + bob * 0.8, radius * 0.20, 0, Math.PI * 2);
    ctx.fill();

    // 5. Human Head & Face
    ctx.fillStyle = skinColor;
    ctx.beginPath();
    ctx.arc(0, -radius * 0.42, radius * 0.40, 0, Math.PI * 2);
    ctx.fill();

    // Hair / Sleek Cap
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(0, -radius * 0.52, radius * 0.40, Math.PI, Math.PI * 2);
    ctx.fill();

    // High-Tech Cyber Visor (Eyes)
    const lookX = lastDir ? lastDir.x * 2.5 : 0;
    const lookY = lastDir ? lastDir.y * 2.0 : 0;
    ctx.fillStyle = visorColor;
    ctx.beginPath();
    ctx.roundRect(-radius * 0.30 + lookX, -radius * 0.48 + lookY, radius * 0.60, radius * 0.22, 3);
    ctx.fill();

    ctx.restore();
  }

  // Draw Mythic Devil / Demon Hunter AI
  drawDevilHunter(ctx, cx, cy, radius, level, isStunned, animTime, isIntercepting) {
    ctx.save();
    ctx.translate(cx, cy);

    const demonColor = level === 3 ? '#b91c1c' : (isIntercepting ? '#c2410c' : '#dc2626');
    const darkDemon = '#7f1d1d';
    const hornColor = '#0f172a';
    const eyeColor = isStunned ? '#38bdf8' : '#fbbf24';

    // 1. Devil Aura Flame Glow
    const auraPulse = Math.sin(animTime * 8) * 2;
    const auraGrad = ctx.createRadialGradient(0, 0, radius * 0.3, 0, 0, radius * 1.35 + auraPulse);
    auraGrad.addColorStop(0, level === 3 ? 'rgba(239, 68, 68, 0.35)' : 'rgba(220, 38, 38, 0.22)');
    auraGrad.addColorStop(1, 'transparent');
    ctx.fillStyle = auraGrad;
    ctx.beginPath();
    ctx.arc(0, 0, radius * 1.35, 0, Math.PI * 2);
    ctx.fill();

    // 2. Demon Wings / Spiky Cloak
    const flap = Math.sin(animTime * 10) * 2.5;
    ctx.fillStyle = darkDemon;
    // Left Wing
    ctx.beginPath();
    ctx.moveTo(-radius * 0.3, 0);
    ctx.quadraticCurveTo(-radius * 1.25, -radius * 0.5 + flap, -radius * 1.05, radius * 0.6);
    ctx.lineTo(-radius * 0.3, radius * 0.4);
    ctx.closePath();
    ctx.fill();

    // Right Wing
    ctx.beginPath();
    ctx.moveTo(radius * 0.3, 0);
    ctx.quadraticCurveTo(radius * 1.25, -radius * 0.5 + flap, radius * 1.05, radius * 0.6);
    ctx.lineTo(radius * 0.3, radius * 0.4);
    ctx.closePath();
    ctx.fill();

    // 3. Sharp Curved Demon Horns
    ctx.fillStyle = hornColor;
    // Left Horn
    ctx.beginPath();
    ctx.moveTo(-radius * 0.4, -radius * 0.2);
    ctx.quadraticCurveTo(-radius * 0.95, -radius * 0.8, -radius * 0.65, -radius * 1.25);
    ctx.quadraticCurveTo(-radius * 0.35, -radius * 0.7, -radius * 0.15, -radius * 0.4);
    ctx.closePath();
    ctx.fill();

    // Right Horn
    ctx.beginPath();
    ctx.moveTo(radius * 0.4, -radius * 0.2);
    ctx.quadraticCurveTo(radius * 0.95, -radius * 0.8, radius * 0.65, -radius * 1.25);
    ctx.quadraticCurveTo(radius * 0.35, -radius * 0.7, radius * 0.15, -radius * 0.4);
    ctx.closePath();
    ctx.fill();

    // 4. Demon Head & Fierce Jaw
    ctx.fillStyle = demonColor;
    ctx.beginPath();
    ctx.moveTo(0, -radius * 0.55); // Top forehead
    ctx.lineTo(radius * 0.65, -radius * 0.12); // Right cheek
    ctx.lineTo(radius * 0.50, radius * 0.48); // Right jaw
    ctx.lineTo(0, radius * 0.92); // Pointy Chin
    ctx.lineTo(-radius * 0.50, radius * 0.48); // Left jaw
    ctx.lineTo(-radius * 0.65, -radius * 0.12); // Left cheek
    ctx.closePath();
    ctx.fill();

    // Inner facial shading
    ctx.fillStyle = darkDemon;
    ctx.beginPath();
    ctx.moveTo(-radius * 0.35, radius * 0.2);
    ctx.lineTo(0, radius * 0.75);
    ctx.lineTo(radius * 0.35, radius * 0.2);
    ctx.lineTo(0, radius * 0.4);
    ctx.closePath();
    ctx.fill();

    // 5. Piercing Slanted Demon Eyes
    ctx.fillStyle = eyeColor;
    // Left Eye
    ctx.beginPath();
    ctx.moveTo(-radius * 0.45, -radius * 0.15);
    ctx.lineTo(-radius * 0.15, -radius * 0.05);
    ctx.lineTo(-radius * 0.25, 0.06);
    ctx.closePath();
    ctx.fill();

    // Right Eye
    ctx.beginPath();
    ctx.moveTo(radius * 0.45, -radius * 0.15);
    ctx.lineTo(radius * 0.15, -radius * 0.05);
    ctx.lineTo(radius * 0.25, 0.06);
    ctx.closePath();
    ctx.fill();

    // Demon Pupils
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(-radius * 0.28, -radius * 0.04, radius * 0.07, 0, Math.PI * 2);
    ctx.arc(radius * 0.28, -radius * 0.04, radius * 0.07, 0, Math.PI * 2);
    ctx.fill();

    // 6. Evil Fangs
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.moveTo(-radius * 0.20, radius * 0.32);
    ctx.lineTo(-radius * 0.12, radius * 0.52);
    ctx.lineTo(-radius * 0.04, radius * 0.32);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(radius * 0.20, radius * 0.32);
    ctx.lineTo(radius * 0.12, radius * 0.52);
    ctx.lineTo(radius * 0.04, radius * 0.32);
    ctx.fill();

    // Stun / Frozen halo effect
    if (isStunned) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(0, -radius * 0.6, radius * 0.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  loop() {
    let lastTime = performance.now();
    const frame = (time) => {
      const dt = Math.min(0.1, (time - lastTime) / 1000.0);
      lastTime = time;

      this.update(dt);
      this.draw();

      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }
}

// Start Game on load
window.addEventListener('DOMContentLoaded', () => {
  window.gameApp = new AIMazeHunterApp();
});
