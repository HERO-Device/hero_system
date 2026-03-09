"""
Memory Game — spatial memory assessment.

Shows a grid of icons for a fixed encoding period (8 seconds), then
prompts the patient to recall where each icon was located. Tracks score,
accuracy, and per-trial cell distance for downstream analysis.

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
    """
    Spatial icon memory game for the HERO consultation system.

    Displays a 4×2 grid of icons during encoding, clears the screen, then
    prompts the patient to recall each icon's position one at a time.
    Provides colour-coded feedback per trial and compiles a results dict.

    Attributes:
        parent: Parent Consultation instance, or None in standalone mode.
        auto_run: If True, skip instruction screens (encoding still runs).
        running: Main loop control flag.
        icons_folder: Path to the folder containing PNG icon images.
        display_size: Vector2 dimensions of the display area.
        display_screen: DisplayScreen for the upper screen.
        touch_screen: TouchScreen for the lower screen.
        title_font: Font for large title text.
        instruction_font: Font for instruction text.
        score_font: Font for score display.
        icons: List of loaded and scaled icon Surfaces.
        phase: Current game phase string — 'waiting', 'encoding', 'recall', or 'feedback'.
        symbol_positions: Dict mapping icon index to (row, col) grid cell.
        recalled_symbols: Set of icon indices already recalled.
        incorrect_symbols: Set of icon indices recalled incorrectly.
        current_symbol_index: Index into symbols_to_recall for the current prompt.
        symbols_to_recall: Shuffled list of icon indices to recall.
        score: Count of correctly recalled icons.
        total_trials: Total number of recall attempts so far.
        encoding_start_time: pg.time.get_ticks() value when encoding began.
        feedback_start_time: pg.time.get_ticks() value when feedback began.
        correct_pos: (row, col) of the correct cell for the last trial.
        wrong_pos: (row, col) of the clicked cell if incorrect, or None.
        results: Dict populated by exit_sequence with summary statistics.
        trial_log: List of per-trial dicts with correctness and distance data.
    """

    def __init__(self, parent=None, auto_run=False,
                 icons_folder="hero/consultation/resources/graphics/games/memory_icons"):
        """
        Initialise the Memory Game and load icon images.

        Args:
            parent: Parent Consultation instance. If provided, screens are shared.
            auto_run: If True, skip instruction wait loops.
            icons_folder: Path to folder containing PNG icon images.
        """
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

        # Results
        self.results = {}
        self.trial_log = []

    def _load_icons(self):
        """
        Load and scale icon images from the icons folder.

        Returns:
            List of pg.Surface objects scaled to ICON_SIZE × ICON_SIZE.
            Returns an empty list if the folder does not exist.
        """
        icons = []

        if not os.path.exists(self.icons_folder):
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
                pass

        return icons

    def _start_game(self):
        """Randomly assign icons to grid cells and begin the encoding phase."""
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
        """
        Compute the pixel rect for a grid cell in bottom-screen coordinates.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            pg.Rect for the cell in bottom-screen pixel coordinates.
        """
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
        """
        Convert a bottom-screen mouse position to a grid cell.

        Args:
            pos: (x, y) position in bottom-screen pixel coordinates.

        Returns:
            (row, col) tuple of the clicked cell, or None if outside the grid.
        """
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                if self._get_cell_rect(row, col).collidepoint(pos):
                    return (row, col)
        return None

    def _handle_recall_click(self, clicked_cell):
        """
        Process a recall attempt for the current icon prompt.

        Compares the clicked cell against the correct cell, updates score and
        feedback state, and advances to the feedback phase.

        Args:
            clicked_cell: (row, col) tuple of the grid cell the patient tapped.
        """
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
        """
        Advance game state based on elapsed time.

        Transitions from encoding → recall after ENCODING_TIME ms, and from
        feedback → recall after 1.5 s. Ends the game when all icons are recalled.
        """
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

    def _draw_grid(self, surface):
        """
        Render the icon grid onto a surface.

        During encoding, all icons are shown. During recall/feedback, only
        recalled icons are shown with colour-coded feedback.

        Args:
            surface: pygame Surface to draw onto.
        """
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
        """
        Draw the current icon prompt and running score on the bottom screen.

        Args:
            surface: pygame Surface to draw onto.
        """
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
        """Render the current game state to both physical screens."""
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

    def _instruction_loop(self):
        """
        Show the instruction screen and wait for the patient to press Start.

        In auto_run mode, starts the game immediately without waiting.
        """
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
        """Show the instruction screen and initialise the game."""
        self.running = True
        self.phase = "waiting"
        self._instruction_loop()

    def exit_sequence(self):
        """
        Compile per-trial results into self.results.

        Computes accuracy percentage and average Manhattan cell distance.
        """
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

    def loop(self):
        """
        Main game loop — called by the orchestrator.

        Processes input events, updates game state each frame, and renders
        until all icons have been recalled or the loop is exited.
        """
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
