import pygame
import math
from typing import Dict, List, Tuple, Optional

class UI:
    def __init__(self, screen_width: int, screen_height: int, sidebar_width: int = 340):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.sidebar_width = sidebar_width
        self.maze_view_width = screen_width - sidebar_width
        
        # Color Palette - Cyberpunk / Dark Neomorphism
        self.BG_COLOR = (13, 17, 23)
        self.PANEL_BG = (22, 27, 34)
        self.PANEL_BORDER = (48, 54, 61)
        self.TEXT_PRIMARY = (240, 246, 252)
        self.TEXT_SECONDARY = (139, 148, 158)
        self.ACCENT_CYAN = (56, 189, 248)
        self.ACCENT_PURPLE = (168, 85, 247)
        self.ACCENT_GREEN = (46, 204, 113)
        self.ACCENT_YELLOW = (241, 196, 15)
        self.ACCENT_RED = (239, 68, 68)

        # Initialize Fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 22, bold=True)
        self.font_section = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 16, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 13)
        self.font_mono = pygame.font.SysFont("Consolas, Courier New, monospace", 12)
        self.font_huge = pygame.font.SysFont("Segoe UI, Arial, sans-serif", 36, bold=True)

    def draw_panel_box(self, surface: pygame.Surface, rect: pygame.Rect, title: str = "", border_color=None):
        if border_color is None:
            border_color = self.PANEL_BORDER
        # Background
        pygame.draw.rect(surface, self.PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(surface, border_color, rect, 1, border_radius=8)
        
        if title:
            title_surf = self.font_section.render(title, True, self.TEXT_PRIMARY)
            surface.blit(title_surf, (rect.x + 12, rect.y + 10))
            # Header line
            pygame.draw.line(
                surface, self.PANEL_BORDER,
                (rect.x + 10, rect.y + 32),
                (rect.x + rect.width - 10, rect.y + 32),
                1
            )

    def draw_sidebar(
        self,
        surface: pygame.Surface,
        level: int,
        score: int,
        orbs_left: int,
        total_orbs: int,
        astar_data: Dict,
        cluster_data: Dict,
        predictor_data: Dict,
        prob_dict: Dict[str, float],
        toggles: Dict[str, bool],
        player_emp: int,
        player_decoy: bool
    ):
        sidebar_rect = pygame.Rect(self.maze_view_width, 0, self.sidebar_width, self.screen_height)
        pygame.draw.rect(surface, (18, 22, 29), sidebar_rect)
        pygame.draw.line(surface, self.PANEL_BORDER, (self.maze_view_width, 0), (self.maze_view_width, self.screen_height), 2)

        cur_y = 12
        margin_x = self.maze_view_width + 12
        card_w = self.sidebar_width - 24

        # 1. Level Header Card
        header_h = 75
        h_rect = pygame.Rect(margin_x, cur_y, card_w, header_h)
        level_colors = {1: self.ACCENT_CYAN, 2: self.ACCENT_GREEN, 3: self.ACCENT_PURPLE}
        level_names = {1: "LEVEL 1: A* SEARCH", 2: "LEVEL 2: K-MEANS ZONES", 3: "LEVEL 3: PREDICTIVE AI"}
        
        self.draw_panel_box(surface, h_rect, border_color=level_colors.get(level, self.ACCENT_CYAN))
        
        lvl_title = self.font_title.render(level_names.get(level, f"LEVEL {level}"), True, level_colors.get(level, self.ACCENT_CYAN))
        surface.blit(lvl_title, (margin_x + 12, cur_y + 8))

        stats_txt = f"Score: {score}  |  Orbs: {total_orbs - orbs_left}/{total_orbs}  |  EMP: {player_emp}"
        s_surf = self.font_body.render(stats_txt, True, self.TEXT_PRIMARY)
        surface.blit(s_surf, (margin_x + 12, cur_y + 42))

        cur_y += header_h + 10

        # 2. A* Pathfinding Telemetry Box
        astar_h = 95
        a_rect = pygame.Rect(margin_x, cur_y, card_w, astar_h)
        self.draw_panel_box(surface, a_rect, "1. A* SEARCH TELEMETRY")
        
        nodes_txt = f"Nodes Visited: {astar_data.get('nodes_visited_count', 0)}"
        time_txt = f"Compute Time: {astar_data.get('execution_time_ms', 0.0):.2f} ms"
        cost_txt = f"Path Steps: {len(astar_data.get('path', []))} (Cost: {astar_data.get('cost', 0.0):.1f})"
        
        surface.blit(self.font_mono.render(nodes_txt, True, self.ACCENT_CYAN), (margin_x + 14, cur_y + 36))
        surface.blit(self.font_mono.render(time_txt, True, self.TEXT_PRIMARY), (margin_x + 14, cur_y + 54))
        surface.blit(self.font_mono.render(cost_txt, True, self.TEXT_SECONDARY), (margin_x + 14, cur_y + 72))

        cur_y += astar_h + 10

        # 3. Unsupervised K-Means Telemetry Box
        unsup_h = 120
        u_rect = pygame.Rect(margin_x, cur_y, card_w, unsup_h)
        u_border = self.ACCENT_GREEN if level >= 2 else self.PANEL_BORDER
        self.draw_panel_box(surface, u_rect, "2. UNSUPERVISED (K-MEANS)", border_color=u_border)

        if level < 2:
            locked_surf = self.font_body.render("🔒 Locked in Level 1 (Press '2' to unlock)", True, self.TEXT_SECONDARY)
            surface.blit(locked_surf, (margin_x + 14, cur_y + 50))
        else:
            pcts = cluster_data.get("percentages", [33.3, 33.3, 33.3])
            inertia = cluster_data.get("inertia", 0.0)
            
            # Cluster Breakdown Bar
            bar_w = card_w - 28
            bar_y = cur_y + 40
            bar_h = 12
            
            x_offset = margin_x + 14
            colors = [self.ACCENT_GREEN, self.ACCENT_YELLOW, self.ACCENT_RED]
            labels = ["Safe", "Transit", "Hotspot"]
            
            for i, p in enumerate(pcts[:3]):
                seg_w = max(2, int(bar_w * (p / 100.0)))
                pygame.draw.rect(surface, colors[i], (x_offset, bar_y, seg_w, bar_h), border_radius=2)
                x_offset += seg_w

            # Legend
            legend_y = bar_y + 18
            for i in range(3):
                lx = margin_x + 14 + i * 95
                pygame.draw.circle(surface, colors[i], (lx + 4, legend_y + 7), 4)
                lbl_surf = self.font_mono.render(f"{labels[i]}: {pcts[i]:.0f}%", True, self.TEXT_PRIMARY)
                surface.blit(lbl_surf, (lx + 12, legend_y))

            inertia_txt = f"K-Means Inertia: {inertia:.1f} | Clusters: K=3"
            surface.blit(self.font_mono.render(inertia_txt, True, self.TEXT_SECONDARY), (margin_x + 14, legend_y + 24))

        cur_y += unsup_h + 10

        # 4. Supervised Learning Telemetry Box
        sup_h = 160
        s_rect = pygame.Rect(margin_x, cur_y, card_w, sup_h)
        s_border = self.ACCENT_PURPLE if level >= 3 else self.PANEL_BORDER
        self.draw_panel_box(surface, s_rect, "3. SUPERVISED (PREDICTOR)", border_color=s_border)

        if level < 3:
            locked_surf = self.font_body.render("🔒 Locked in Level 1 & 2 (Press '3' to unlock)", True, self.TEXT_SECONDARY)
            surface.blit(locked_surf, (margin_x + 14, cur_y + 60))
        else:
            pred_dir = predictor_data.get("last_predicted_dir", "N/A")
            samples = predictor_data.get("samples_count", 0)
            conf = predictor_data.get("last_confidence", 0.0)
            acc = predictor_data.get("live_accuracy", 0.0)

            # Prediction status
            p_str = f"Predicting Next Move: {pred_dir} ({conf:.0f}%)"
            surface.blit(self.font_body.render(p_str, True, self.ACCENT_PURPLE), (margin_x + 14, cur_y + 36))
            
            acc_str = f"Samples: {samples} | Online Accuracy: {acc:.1f}%"
            surface.blit(self.font_mono.render(acc_str, True, self.TEXT_SECONDARY), (margin_x + 14, cur_y + 56))

            # Direction probability mini-bars
            bar_start_y = cur_y + 78
            dir_keys = ["UP [^]", "DOWN [v]", "LEFT [<]", "RIGHT [>]"]
            for idx, d_name in enumerate(dir_keys):
                p_val = prob_dict.get(d_name, 0.0)
                by = bar_start_y + idx * 18
                
                # Label
                surface.blit(self.font_mono.render(d_name, True, self.TEXT_PRIMARY), (margin_x + 14, by))
                
                # Progress Bar
                p_bar_x = margin_x + 75
                p_bar_w = card_w - 145
                pygame.draw.rect(surface, (40, 45, 55), (p_bar_x, by + 3, p_bar_w, 8), border_radius=3)
                fill_w = int(p_bar_w * p_val)
                if fill_w > 0:
                    pygame.draw.rect(surface, self.ACCENT_PURPLE, (p_bar_x, by + 3, fill_w, 8), border_radius=3)
                
                # % Text
                pct_str = f"{p_val * 100.0:3.0f}%"
                surface.blit(self.font_mono.render(pct_str, True, self.TEXT_SECONDARY), (p_bar_x + p_bar_w + 8, by))

        cur_y += sup_h + 10

        # 5. Hotkeys & Toggles Card
        ctrl_h = self.screen_height - cur_y - 12
        if ctrl_h > 80:
            c_rect = pygame.Rect(margin_x, cur_y, card_w, ctrl_h)
            self.draw_panel_box(surface, c_rect, "CONTROLS & VISUALIZERS")

            hotkeys = [
                f"[V] Explored Nodes: {'ON' if toggles.get('explored', False) else 'OFF'}",
                f"[H] Zone Heatmap: {'ON' if toggles.get('heatmap', False) else 'OFF'}",
                f"[P] Intercept Ghost: {'ON' if toggles.get('ghost', True) else 'OFF'}",
                "[SPACE] Decoy | [E] EMP Shockwave",
                "[1/2/3] Switch Level | [R] New Maze",
                "[TAB] Deep ML Guide"
            ]
            
            for i, hk in enumerate(hotkeys):
                hy = cur_y + 36 + i * 18
                if hy + 16 < cur_y + ctrl_h:
                    surface.blit(self.font_mono.render(hk, True, self.TEXT_PRIMARY if i < 3 else self.TEXT_SECONDARY), (margin_x + 14, hy))

    def draw_deep_ml_modal(self, surface: pygame.Surface):
        """Full-screen overlay explaining the mathematical and engineering story."""
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((10, 14, 20, 240))
        surface.blit(overlay, (0, 0))

        modal_w = min(880, self.screen_width - 80)
        modal_h = min(620, self.screen_height - 60)
        modal_rect = pygame.Rect((self.screen_width - modal_w) // 2, (self.screen_height - modal_h) // 2, modal_w, modal_h)

        pygame.draw.rect(surface, (22, 27, 34), modal_rect, border_radius=12)
        pygame.draw.rect(surface, self.ACCENT_CYAN, modal_rect, 2, border_radius=12)

        title = self.font_title.render("🧠 AI MAZE HUNTER — MACHINE LEARNING ARCHITECTURE", True, self.ACCENT_CYAN)
        surface.blit(title, (modal_rect.x + 24, modal_rect.y + 20))

        subtitle = self.font_body.render("How Supervised, Unsupervised & Graph Search Algorithms collaborate in real-time:", True, self.TEXT_SECONDARY)
        surface.blit(subtitle, (modal_rect.x + 24, modal_rect.y + 50))

        # 3 Pillar Columns
        col_w = (modal_w - 68) // 3
        
        # Pillar 1: A* Search
        p1_rect = pygame.Rect(modal_rect.x + 20, modal_rect.y + 80, col_w, modal_h - 140)
        self.draw_panel_box(surface, p1_rect, "1. A* GRAPH SEARCH", border_color=self.ACCENT_CYAN)
        p1_text = [
            "• Role: Optimal Pathfinding",
            "• Evaluation Function:",
            "    f(n) = g(n) + h(n)",
            "• g(n): Exact cost from",
            "  Hunter to current node",
            "• h(n): Manhattan heuristic",
            "  h(n) = |x1-x2| + |y1-y2|",
            "• Dynamic Cost Modulation:",
            "  Zone cost multipliers",
            "  alter g(n) edge weights!"
        ]
        for idx, line in enumerate(p1_text):
            surface.blit(self.font_mono.render(line, True, self.TEXT_PRIMARY), (p1_rect.x + 10, p1_rect.y + 40 + idx * 20))

        # Pillar 2: Unsupervised Learning
        p2_rect = pygame.Rect(modal_rect.x + 30 + col_w, modal_rect.y + 80, col_w, modal_h - 140)
        self.draw_panel_box(surface, p2_rect, "2. UNSUPERVISED (K-MEANS)", border_color=self.ACCENT_GREEN)
        p2_text = [
            "• Role: Map Zone Segmentation",
            "• K-Means Algorithm (K=3):",
            "  Minimizes cluster inertia",
            "• Multi-factor Telemetry:",
            "  - Visit Frequency",
            "  - Dwell Duration",
            "  - Hunter Proximity Hits",
            "  - Topology Connectivity",
            "• Zones: Safe (Green),",
            "  Transit (Yellow),",
            "  Danger Hotspot (Red)"
        ]
        for idx, line in enumerate(p2_text):
            surface.blit(self.font_mono.render(line, True, self.TEXT_PRIMARY), (p2_rect.x + 10, p2_rect.y + 40 + idx * 20))

        # Pillar 3: Supervised Learning
        p3_rect = pygame.Rect(modal_rect.x + 40 + col_w * 2, modal_rect.y + 80, col_w, modal_h - 140)
        self.draw_panel_box(surface, p3_rect, "3. SUPERVISED PREDICTOR", border_color=self.ACCENT_PURPLE)
        p3_text = [
            "• Role: Player Trajectory Prediction",
            "• Random Forest & Markov:",
            "  Ensemble classification",
            "• Input Features:",
            "  - Wall radar sensors",
            "  - Vector to Hunter",
            "  - Direction history (lag-k)",
            "• Lookahead Interception:",
            "  Feeds predicted cell (Tx, Ty)",
            "  directly into A* target!"
        ]
        for idx, line in enumerate(p3_text):
            surface.blit(self.font_mono.render(line, True, self.TEXT_PRIMARY), (p3_rect.x + 10, p3_rect.y + 40 + idx * 20))

        # Footer
        close_txt = self.font_body.render("Press [TAB] or [ESC] to return to the game", True, self.ACCENT_YELLOW)
        surface.blit(close_txt, (modal_rect.x + modal_w // 2 - close_txt.get_width() // 2, modal_rect.y + modal_h - 35))

    def draw_banner_modal(self, surface: pygame.Surface, title: str, subtitle: str, color, button_prompt: str = "Press [ENTER] to Continue"):
        overlay = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((10, 14, 20, 200))
        surface.blit(overlay, (0, 0))

        b_width = 540
        b_height = 220
        b_rect = pygame.Rect((self.screen_width - b_width) // 2, (self.screen_height - b_height) // 2, b_width, b_height)

        pygame.draw.rect(surface, (22, 27, 34), b_rect, border_radius=12)
        pygame.draw.rect(surface, color, b_rect, 2, border_radius=12)

        t_surf = self.font_huge.render(title, True, color)
        s_surf = self.font_body.render(subtitle, True, self.TEXT_PRIMARY)
        btn_surf = self.font_section.render(button_prompt, True, self.ACCENT_YELLOW)

        surface.blit(t_surf, (b_rect.centerx - t_surf.get_width() // 2, b_rect.y + 30))
        surface.blit(s_surf, (b_rect.centerx - s_surf.get_width() // 2, b_rect.y + 90))
        surface.blit(btn_surf, (b_rect.centerx - btn_surf.get_width() // 2, b_rect.y + 150))
