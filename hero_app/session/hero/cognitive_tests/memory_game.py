"""
Memory Game - Spatial memory assessment

Shows a 3x3 grid of icons for 8 seconds (encoding phase).
Patient then recalls where each icon was located (recall phase).
Tracks score and response accuracy.

Adapted from standalone version to work within the HERO consultation framework.
"""

import os
import random
import pygame as pg

from hero.consultation.display_screen import DisplayScreen
from hero.consultation.touch_screen import TouchScreen, GameObjects, GameButton
from hero.consultation.screen import Colours

# Constants
GRID_COLS = 4
GRID_ROWS = 2
GRID_SIZE  = GRID_COLS * GRID_ROWS
CELL_SIZE = 140
GRID_PADDING = 15
ENCODING_TIME = 8000  # ms
ICON_SIZE = 100

# Colors
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (158, 158, 158)
GREEN = (76, 175, 80)
RED = (244, 67, 54)
BLACK = (0, 0, 0)
TEXT_GRAY = (100, 100, 100)


class MemoryGame:
    def __init__(self, parent=None, auto_run=False,
                 icons_folder="hero/consultation/resources/graphics/games/memory_icons"):
        self.parent = parent
        self.auto_run = auto_run
        self.running = False
        self.icons_folder = icons_folder

        # Set up screens from parent or create standalone
        if parent is not None:
            self.display_size = parent.display_size
            self.top_screen = parent.top_screen
            self.bottom_screen = parent.bottom_screen
            self.display_screen = DisplayScreen(self.display_size, avatar=parent.avatar)
        else:
            self.display_size = pg.Vector2(500, 600)
            self.window = pg.display.set_mode(
                (int(self.display_size.x), int(self.display_size.y * 2))
            )
            self.top_screen = self.window.subsurface(
                (0, 0), (int(self.display_size.x), int(self.display_size.y))
            )
            self.bottom_screen = self.window.subsurface(
                (0, int(self.display_size.y)),
                (int(self.display_size.x), int(self.display_size.y))
            )
            self.display_screen = DisplayScreen(self.display_size)

        self.touch_screen = TouchScreen(self.display_size)

        # Fonts
        self.title_font = pg.font.SysFont('Arial', 28, bold=True)
        self.instruction_font = pg.font.SysFont('Arial', 22)
        self.score_font = pg.font.SysFont('Arial', 20)

        # Load icons
        self.icons = self._load_icons()

        # Game state
        self.phase = "waiting"
        self.symbol_positions = {}
        self.recalled_symbols = set()
        self.incorrect_symbols = set()
        self.current_symbol_index = 0
        self.symbols_to_recall = []
        self.score = 0
        self.total_trials = 0
        self.encoding_start_time = 0
        self.feedback_start_time = 0
        self.correct_pos = None
        self.wrong_pos = None
        self.button_rect = None

        # Grid centred on bottom screen
        w = int(self.display_size.x)
        grid_total_width = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GRID_PADDING

        # Results
        self.results = {}
        self.trial_log = []   # per-trial: {trial_num, icon_idx, correct_cell, clicked_cell, correct, distance_px}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_icons(self):
        """Load icon images from the Icons folder."""
        icons = []

        if not os.path.exists(self.icons_folder):
            print(f"⚠ Memory Game: Icons folder not found at '{self.icons_folder}'")
            return icons

        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        icon_files = sorted([
            f for f in os.listdir(self.icons_folder)
            if f.lower().endswith(image_extensions)
        ])

        for filename in icon_files[:9]:
            filepath = os.path.join(self.icons_folder, filename)
            try:
                image = pg.image.load(filepath).convert_alpha()
                scaled = pg.transform.smoothscale(image, (ICON_SIZE, ICON_SIZE))
                icons.append(scaled)
            except Exception as e:
                print(f"⚠ Memory Game: Could not load {filename}: {e}")

        print(f"✓ Memory Game: Loaded {len(icons)} icons")
        return icons

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def _start_game(self):
        """Initialise a new game round."""
        self.phase = "encoding"
        self.symbol_positions = {}
        self.recalled_symbols = set()
        self.incorrect_symbols = set()
        self.current_symbol_index = 0
        self.score = 0
        self.total_trials = 0
        self.correct_pos = None
        self.wrong_pos = None

        positions = [(row, col) for row in range(GRID_ROWS) for col in range(GRID_COLS)]
        random.shuffle(positions)
        for idx in range(min(len(self.icons), GRID_COLS * GRID_ROWS)):
            self.symbol_positions[idx] = positions[idx]

        self.symbols_to_recall = list(range(min(len(self.icons), GRID_COLS * GRID_ROWS)))
        random.shuffle(self.symbols_to_recall)

        self.encoding_start_time = pg.time.get_ticks()

    def _get_cell_rect(self, row, col):
        grid_total_width = GRID_COLS * CELL_SIZE + (GRID_COLS - 1) * GRID_PADDING
        grid_total_height = GRID_ROWS * CELL_SIZE + (GRID_ROWS - 1) * GRID_PADDING
        w = int(self.display_size.x)
        h = int(self.display_size.y)
        offset_x = (w - grid_total_width) // 2
        offset_y = (h - grid_total_height) // 2 + 40  # slight downward offset for title
        x = offset_x + col * (CELL_SIZE + GRID_PADDING)
        y = offset_y + row * (CELL_SIZE + GRID_PADDING)
        return pg.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def _get_clicked_cell(self, pos):
        """Convert bottom-screen mouse position to grid cell."""
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if self._get_cell_rect(row, col).collidepoint(pos):
                    return (row, col)
        return None

    def _handle_recall_click(self, clicked_cell):
        """Handle a recall attempt."""
        if self.current_symbol_index >= len(self.symbols_to_recall):
            return

        symbol_idx = self.symbols_to_recall[self.current_symbol_index]
        correct_cell = self.symbol_positions[symbol_idx]

        self.total_trials += 1

        is_correct = (clicked_cell == correct_cell)
        if is_correct:
            self.score += 1
            self.recalled_symbols.add(symbol_idx)
            self.correct_pos = correct_cell
            self.wrong_pos = None
        else:
            self.incorrect_symbols.add(symbol_idx)
            self.recalled_symbols.add(symbol_idx)
            self.correct_pos = correct_cell
            self.wrong_pos = clicked_cell

        # Cell distance (Manhattan, in grid units) — 0 means correct
        cell_distance = (abs(clicked_cell[0] - correct_cell[0]) +
                         abs(clicked_cell[1] - correct_cell[1]))

        self.trial_log.append({
            'trial_num': self.total_trials,
            'icon_idx': symbol_idx,
            'correct_cell': list(correct_cell),
            'clicked_cell': list(clicked_cell),
            'correct': is_correct,
            'cell_distance': cell_distance,
        })

        self.phase = "feedback"
        self.feedback_start_time = pg.time.get_ticks()

    def _update(self):
        """Update game state each frame."""
        if self.phase == "encoding":
            if pg.time.get_ticks() - self.encoding_start_time >= ENCODING_TIME:
                self.phase = "recall"

        elif self.phase == "feedback":
            if pg.time.get_ticks() - self.feedback_start_time >= 1500:
                self.current_symbol_index += 1
                self.phase = "recall"
                self.correct_pos = None
                self.wrong_pos = None

        # Auto-complete when all recalled
        elif self.phase == "recall":
            if self.current_symbol_index >= len(self.symbols_to_recall) and self.symbols_to_recall:
                self.running = False

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_grid(self, surface):
        """Draw the icon grid onto the given surface."""
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                rect = self._get_cell_rect(row, col)
                color = DARK_GRAY
                icon_to_draw = None

                if self.phase == "encoding":
                    for icon_idx, (s_row, s_col) in self.symbol_positions.items():
                        if (s_row, s_col) == (row, col):
                            icon_to_draw = self.icons[icon_idx]
                            color = (117, 117, 117)
                            break

                elif self.phase in ("recall", "feedback"):
                    for icon_idx in self.recalled_symbols:
                        s_row, s_col = self.symbol_positions[icon_idx]
                        if (s_row, s_col) == (row, col):
                            icon_to_draw = self.icons[icon_idx]
                            color = RED if icon_idx in self.incorrect_symbols else GREEN
                            break

                    if self.phase == "feedback":
                        if self.correct_pos and (row, col) == self.correct_pos:
                            for icon_idx, (s_row, s_col) in self.symbol_positions.items():
                                if (s_row, s_col) == (row, col):
                                    icon_to_draw = self.icons[icon_idx]
                                    color = RED if icon_idx in self.incorrect_symbols else GREEN
                                    break
                        elif self.wrong_pos and (row, col) == self.wrong_pos:
                            color = RED

                pg.draw.rect(surface, color, rect, border_radius=8)
                pg.draw.rect(surface, WHITE, rect, 2, border_radius=8)

                if icon_to_draw:
                    icon_rect = icon_to_draw.get_rect(center=rect.center)
                    surface.blit(icon_to_draw, icon_rect)

    def _draw_ui(self, surface):
        """Draw icon prompt and score on bottom screen only."""
        w = int(self.display_size.x)
        h = int(self.display_size.y)

        # Icon prompt during recall — show which icon to place
        if self.phase in ("recall", "feedback"):
            if self.current_symbol_index < len(self.symbols_to_recall):
                current_icon = self.icons[self.symbols_to_recall[self.current_symbol_index]]
                icon_rect = current_icon.get_rect(center=(w // 2, 60))
                surface.blit(current_icon, icon_rect)

        # Score
        if self.total_trials > 0:
            pct = int((self.score / self.total_trials) * 100)
            score_text = f"Score: {self.score}/{self.total_trials} ({pct}%)"
        else:
            score_text = "Score: 0/0"
        score_surf = self.score_font.render(score_text, True, TEXT_GRAY)
        surface.blit(score_surf, score_surf.get_rect(center=(w // 2, h - 40)))

    def _update_display(self):
        """Render top and bottom screens."""
        self.display_screen.refresh()
        if self.phase == "encoding":
            elapsed = (pg.time.get_ticks() - self.encoding_start_time) / 1000
            remaining = max(0, 8 - int(elapsed))
            self.display_screen.instruction = f"Remember the icon positions! ({remaining}s)"
        elif self.phase in ("recall", "feedback"):
            if self.current_symbol_index < len(self.symbols_to_recall):
                self.display_screen.instruction = f"Where was this icon? ({self.current_symbol_index + 1}/{len(self.symbols_to_recall)})"
            else:
                self.display_screen.instruction = "Well done!"
        else:
            self.display_screen.instruction = "Memory Game"
        self.top_screen.blit(self.display_screen.get_surface(), (0, 0))

        # Bottom screen — game surface
        game_surface = pg.Surface(
            (int(self.display_size.x), int(self.display_size.y))
        )
        game_surface.fill(WHITE)
        self._draw_grid(game_surface)
        self._draw_ui(game_surface)
        self.bottom_screen.blit(game_surface, (0, 0))

        pg.display.flip()

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    def _instruction_loop(self):
        """Show instruction screen — avatar + title on top, START button on bottom."""
        if self.auto_run:
            self._start_game()
            return

        self.display_screen.state = 1
        self.display_screen.refresh()
        self.display_screen.instruction = None

        info_rect = pg.Rect(0.3 * self.display_size.x, 0, 0.7 * self.display_size.x, 0.8 * self.display_size.y)
        pg.draw.rect(self.display_screen.surface, Colours.white.value, info_rect)
        self.display_screen.add_multiline_text("Memory Game", rect=info_rect.scale_by(0.9, 0.9), font_size=50)

        info_text = ("A grid of icons will appear on the screen. "
                     "Try to remember where each icon is located. "
                     "After a few seconds they will disappear, and you will need to recall where each one was.")
        self.display_screen.add_multiline_text(rect=info_rect.scale_by(0.9, 0.9), text=info_text, center_vertical=True)

        # Exactly the same button as Shapes
        button_rect = pg.Rect((self.display_size - pg.Vector2(300, 200)) / 2, (300, 200))
        start_button = GameButton(position=button_rect.topleft, size=button_rect.size, text="START", id=1)
        self.touch_screen.sprites = GameObjects([start_button])

        self.top_screen.blit(self.display_screen.get_surface(), (0, 0))
        self.bottom_screen.blit(self.touch_screen.get_surface(), (0, 0))
        pg.display.flip()

        if self.parent and self.parent.config.speech:
            self.parent.speak_text(info_text, visual=True,
                                   display_screen=self.display_screen,
                                   touch_screen=self.touch_screen)

        wait = True
        while wait:
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONDOWN:
                    raw = pg.Vector2(pg.mouse.get_pos())
                    pos = raw - pg.Vector2(0, self.display_size.y)
                    if self.touch_screen.click_test(pos) is not None:
                        wait = False
                elif event.type == pg.QUIT:
                    wait = False

        self.touch_screen.kill_sprites()
        self.display_screen.state = 0
        self.display_screen.refresh()
        self._start_game()

    def entry_sequence(self):
        """Show instructions then start the game."""
        self.running = True
        self.phase = "waiting"

        if self.parent and self.parent.config.speech:
            pass  # speech handled inside _instruction_loop via speak_text

        self._instruction_loop()

    def exit_sequence(self):
        """Compile and store results."""
        accuracy = (
            round((self.score / self.total_trials) * 100, 1)
            if self.total_trials > 0 else 0
        )
        avg_distance = (
            round(sum(t['cell_distance'] for t in self.trial_log) / len(self.trial_log), 2)
            if self.trial_log else 0
        )
        self.results = {
            'score': self.score,
            'total_trials': self.total_trials,
            'accuracy_percent': accuracy,
            'incorrect_count': len(self.incorrect_symbols),
            'avg_cell_distance': avg_distance,   # 0 = perfect, higher = further off
            'trial_log': self.trial_log,
        }
        print(f"✓ Memory Game complete — {self.score}/{self.total_trials} ({accuracy}%)")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def loop(self):
        """Main game loop — called by the orchestrator."""
        self.entry_sequence()
        clock = pg.time.Clock()

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.running = False

                elif event.type == pg.MOUSEBUTTONDOWN:
                    # Convert to bottom-screen coordinates
                    raw = pg.Vector2(pg.mouse.get_pos())
                    pos = raw - pg.Vector2(0, self.display_size.y)

                    if self.phase == "recall":
                        cell = self._get_clicked_cell(pos)
                        if cell is not None:
                            self._handle_recall_click(cell)

            self._update()
            self._update_display()
            clock.tick(60)

        self.exit_sequence()
        
