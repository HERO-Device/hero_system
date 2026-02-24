"""
Trail Making Test (TMT) - Medical Assessment Game

Patient clicks numbered circles in order (1 → 2 → 3 ...).
Tracks completion time, errors, path efficiency, smoothness and pauses.

Supports two input modes:
- MOUSE: click-based (laptop testing)
- TOUCHSCREEN: continuous stylus drawing (Pi touchscreen)

Adapted from standalone version to work within the HERO consultation framework.
"""

import math
import random
import time

import pygame as pg

from hero.consultation.display_screen import DisplayScreen
from hero.consultation.touch_screen import TouchScreen, GameObjects, GameButton
from hero.consultation.screen import Colours

# Configuration
NUM_CIRCLES = 10
FPS = 60

DIFFICULTY_SETTINGS = {
    'small':  {'radius': 20, 'label': 'Hard'},
    'medium': {'radius': 30, 'label': 'Standard'},
    'large':  {'radius': 45, 'label': 'Easy'},
}

PAUSE_THRESHOLD = 1.5       # seconds before a stop counts as a pause
MIN_MOVEMENT_SPEED = 5      # px/s below this = paused
SMOOTHNESS_WINDOW = 10

# Colors
WHITE      = (255, 255, 255)
BLACK      = (0,   0,   0)
GRAY       = (200, 200, 200)
DARK_GRAY  = (100, 100, 100)
GREEN      = (0,   200, 0)
RED        = (255, 0,   0)
BLUE       = (0,   100, 255)
LIGHT_BLUE = (150, 200, 255)
ORANGE     = (255, 165, 0)

MODE_STYLUS = 'touchscreen'
MODE_MOUSE  = 'mouse'


# ------------------------------------------------------------------
# Helper classes (unchanged from original)
# ------------------------------------------------------------------

class TMTCircle:
    def __init__(self, x, y, label, radius):
        self.x = x
        self.y = y
        self.label = label
        self.radius = radius
        self.completed = False
        self.entry_time = None
        self.entry_position = None

    def draw(self, surface, is_current=False):
        color = GREEN if self.completed else (LIGHT_BLUE if is_current else WHITE)
        pg.draw.circle(surface, color, (self.x, self.y), self.radius)
        pg.draw.circle(surface, BLACK, (self.x, self.y), self.radius, 2)
        font = pg.font.Font(None, 36)
        text = font.render(str(self.label), True, BLACK)
        surface.blit(text, text.get_rect(center=(self.x, self.y)))

    def contains_point(self, pos):
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        return math.sqrt(dx * dx + dy * dy) <= self.radius


class PerformanceMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.path_points = []
        self.optimal_distance = 0
        self.actual_distance = 0
        self.pauses = []
        self.current_pause_start = None
        self.last_movement_time = None
        self.movement_speeds = []

    def add_path_point(self, pos, timestamp):
        self.path_points.append({'position': pos, 'timestamp': timestamp})

        if len(self.path_points) > 1:
            prev = self.path_points[-2]
            dx = pos[0] - prev['position'][0]
            dy = pos[1] - prev['position'][1]
            distance = math.sqrt(dx * dx + dy * dy)
            time_delta = timestamp - prev['timestamp']

            if time_delta > 0:
                speed = distance / time_delta
                self.movement_speeds.append(speed)

                if speed < MIN_MOVEMENT_SPEED:
                    if self.current_pause_start is None:
                        self.current_pause_start = timestamp
                else:
                    if self.current_pause_start is not None:
                        pause_duration = timestamp - self.current_pause_start
                        if pause_duration >= PAUSE_THRESHOLD:
                            self.pauses.append({
                                'start': self.current_pause_start,
                                'duration': pause_duration
                            })
                        self.current_pause_start = None
                self.last_movement_time = timestamp

    def calculate_actual_distance(self):
        total = 0
        for i in range(1, len(self.path_points)):
            prev = self.path_points[i - 1]['position']
            curr = self.path_points[i]['position']
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def calculate_optimal_distance(self, circles):
        total = 0
        for i in range(1, len(circles)):
            dx = circles[i].x - circles[i - 1].x
            dy = circles[i].y - circles[i - 1].y
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def calculate_path_efficiency(self):
        if self.actual_distance == 0:
            return 100.0
        return min(100.0, (self.optimal_distance / self.actual_distance) * 100)

    def calculate_path_smoothness(self):
        if len(self.path_points) < SMOOTHNESS_WINDOW:
            return 100.0

        direction_changes = []
        for i in range(SMOOTHNESS_WINDOW, len(self.path_points)):
            points = self.path_points[i - SMOOTHNESS_WINDOW:i]
            angles = []
            for j in range(1, len(points) - 1):
                p1 = points[j - 1]['position']
                p2 = points[j]['position']
                p3 = points[j + 1]['position']
                v1x, v1y = p2[0] - p1[0], p2[1] - p1[1]
                v2x, v2y = p3[0] - p2[0], p3[1] - p2[1]
                mag1 = math.sqrt(v1x * v1x + v1y * v1y)
                mag2 = math.sqrt(v2x * v2x + v2y * v2y)
                if mag1 > 0 and mag2 > 0:
                    cos_angle = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (mag1 * mag2)))
                    angles.append(abs(math.acos(cos_angle)))
            if angles:
                direction_changes.append(sum(angles) / len(angles))

        if not direction_changes:
            return 100.0
        avg = sum(direction_changes) / len(direction_changes)
        return max(0.0, min(100.0, 100 * (1 - avg / math.pi)))

    def get_summary(self, completion_time, completed_circles):
        self.actual_distance = self.calculate_actual_distance()
        self.optimal_distance = self.calculate_optimal_distance(completed_circles)
        speeds = self.movement_speeds
        return {
            'completion_time':  round(completion_time, 2),
            'path_efficiency':  round(self.calculate_path_efficiency(), 2),
            'path_smoothness':  round(self.calculate_path_smoothness(), 2),
            'total_distance':   round(self.actual_distance, 2),
            'optimal_distance': round(self.optimal_distance, 2),
            'pause_count':      len(self.pauses),
            'total_pause_time': round(sum(p['duration'] for p in self.pauses), 2),
            'average_speed':    round(sum(speeds) / len(speeds), 2) if speeds else 0,
            'speed_variability': round(self._std(speeds), 2) if len(speeds) > 1 else 0,
        }

    def _std(self, values):
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class TrailMakingTest:
    def __init__(self, parent=None, auto_run=False,
                 difficulty='medium', test_type='A'):
        self.parent = parent
        self.auto_run = auto_run
        self.running = False
        self.difficulty = difficulty
        self.test_type = test_type
        self.circle_radius = DIFFICULTY_SETTINGS[difficulty]['radius']

        # Set up screens from parent or standalone
        if parent is not None:
            self.display_size = parent.display_size
            self.top_screen = parent.top_screen
            self.bottom_screen = parent.bottom_screen
            self.display_screen = DisplayScreen(self.display_size, avatar=parent.avatar)
        else:
            self.display_size = pg.Vector2(800, 600)
            self.window = pg.display.set_mode(
                (int(self.display_size.x), int(self.display_size.y * 2))
            )
            self.top_screen = self.window.subsurface(
                (0, 0, int(self.display_size.x), int(self.display_size.y))
            )
            self.bottom_screen = self.window.subsurface(
                (0, int(self.display_size.y),
                 int(self.display_size.x), int(self.display_size.y))
            )
            self.display_screen = DisplayScreen(self.display_size)

        self.touch_screen = TouchScreen(self.display_size)

        # Game state
        self.circles = []
        self.current_index = 0
        self.path = []
        self.errors = 0
        self.start_time = None
        self.end_time = None
        self.completed = False
        self.drawing = False
        self.draw_path = []
        self.input_mode = MODE_STYLUS
        self.metrics = PerformanceMetrics()

        # Results
        self.results = {}

        self._generate_circles()

    # ------------------------------------------------------------------
    # Circle generation
    # ------------------------------------------------------------------

    def _generate_circles(self):
        """Generate non-overlapping circles on the bottom screen."""
        self.circles = []
        w = int(self.display_size.x)
        h = int(self.display_size.y)
        margin = self.circle_radius + 50

        attempts = 0
        while len(self.circles) < NUM_CIRCLES and attempts < 1000:
            x = random.randint(margin, w - margin)
            y = random.randint(margin + 60, h - margin - 30)

            valid = all(
                math.sqrt((x - c.x) ** 2 + (y - c.y) ** 2) >= (self.circle_radius + c.radius) * 2
                for c in self.circles
            )
            if valid:
                label = len(self.circles) + 1
                self.circles.append(TMTCircle(x, y, label, self.circle_radius))
            attempts += 1

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _handle_click(self, pos):
        """Mouse-mode: register a click on the bottom screen."""
        if self.completed:
            return

        if self.start_time is None:
            self.start_time = time.time()

        t = time.time() - self.start_time
        expected = self.circles[self.current_index]

        if expected.contains_point(pos):
            expected.completed = True
            expected.entry_time = t
            expected.entry_position = pos

            self.path.append({
                'circle': expected.label,
                'timestamp': t,
                'position': pos,
                'input_method': self.input_mode
            })

            # Add straight-line metrics between clicks
            if self.current_index > 0:
                prev = self.circles[self.current_index - 1]
                self.metrics.add_path_point(prev.entry_position, prev.entry_time)
            self.metrics.add_path_point(pos, t)

            self.current_index += 1
            if self.current_index >= len(self.circles):
                self._complete_test()
        else:
            # Wrong circle clicked
            for c in self.circles:
                if c.contains_point(pos) and not c.completed:
                    self.errors += 1
                    break

    def _handle_touch_movement(self, pos, timestamp):
        """Stylus-mode: handle continuous movement."""
        if self.completed:
            return

        if self.start_time is None:
            self.start_time = time.time()

        self.metrics.add_path_point(pos, timestamp - self.start_time)

        if self.current_index < len(self.circles):
            expected = self.circles[self.current_index]
            if expected.contains_point(pos) and not expected.completed:
                expected.completed = True
                expected.entry_time = timestamp - self.start_time
                expected.entry_position = pos

                self.path.append({
                    'circle': expected.label,
                    'timestamp': expected.entry_time,
                    'position': pos,
                    'input_method': self.input_mode
                })

                self.current_index += 1
                if self.current_index >= len(self.circles):
                    self._complete_test()

    def _complete_test(self):
        """Mark test complete and compile metrics."""
        self.completed = True
        self.end_time = time.time()
        completion_time = self.end_time - self.start_time
        completed_circles = [c for c in self.circles if c.completed]
        self.results = self.metrics.get_summary(completion_time, completed_circles)
        self.results['errors'] = self.errors
        self.results['test_type'] = self.test_type
        self.results['difficulty'] = self.difficulty

        # Raw circle path: each waypoint the patient hit in order
        self.results['circle_path'] = [
            {
                'circle': p['circle'],
                'timestamp_s': round(p['timestamp'], 4),
                'x': p['position'][0],
                'y': p['position'][1],
                'input_method': p['input_method'],
            }
            for p in self.path
        ]

        print(f"✓ Trail Making Test complete — {completion_time:.2f}s, {self.errors} errors")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, surface):
        """Draw the full game onto the given surface."""
        surface.fill(WHITE)
        w = int(self.display_size.x)
        h = int(self.display_size.y)

        font_stats = pg.font.Font(None, 26)

        # Completed connecting lines
        for i in range(1, self.current_index):
            p = self.circles[i - 1]
            c = self.circles[i]
            pg.draw.line(surface, BLUE, (p.x, p.y), (c.x, c.y), 3)

        # Stylus path
        if len(self.draw_path) > 1 and self.input_mode == MODE_STYLUS:
            pg.draw.lines(surface, RED, False, self.draw_path, 2)

        # Line to mouse cursor
        if self.current_index > 0 and not self.completed:
            last = self.circles[self.current_index - 1]
            raw_mouse = pg.mouse.get_pos()
            mouse_on_bottom = (raw_mouse[0], raw_mouse[1] - int(self.display_size.y))
            pg.draw.line(surface, GRAY, (last.x, last.y), mouse_on_bottom, 2)

        # Circles
        for i, circle in enumerate(self.circles):
            circle.draw(surface, is_current=(i == self.current_index))

        # Mode indicator
        mode_font = pg.font.Font(None, 20)
        mode_str = "TOUCH" if self.input_mode == MODE_STYLUS else "MOUSE"
        mode_surf = mode_font.render(f"Mode: {mode_str}", True, WHITE)
        color = ORANGE if self.input_mode == MODE_STYLUS else BLUE
        padding = 8
        box = pg.Rect(
            w - mode_surf.get_width() - padding * 2 - 10, 10,
            mode_surf.get_width() + padding * 2,
            mode_surf.get_height() + padding
        )
        pg.draw.rect(surface, color, box)
        pg.draw.rect(surface, BLACK, box, 2)
        surface.blit(mode_surf, mode_surf.get_rect(center=box.center))

        # Completion overlay — just white out the bottom screen, top screen handles messaging
        if self.completed and self.results:
            overlay = pg.Surface((w, h), pg.SRCALPHA)
            overlay.fill((255, 255, 255, 220))
            surface.blit(overlay, (0, 0))

    def _update_display(self):
        """Render top and bottom screens."""
        self.display_screen.refresh()
        if self.completed:
            self.display_screen.instruction = "Well done!"
        else:
            self.display_screen.instruction = "Trail Making Test"
        self.top_screen.blit(self.display_screen.get_surface(), (0, 0))

        game_surface = pg.Surface(
            (int(self.display_size.x), int(self.display_size.y))
        )
        self._draw(game_surface)
        self.bottom_screen.blit(game_surface, (0, 0))

        pg.display.flip()

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    def _instruction_loop(self):
        """Show instruction screen — avatar + title on top, plain START button on bottom."""
        if self.auto_run:
            return

        self.display_screen.state = 1
        self.display_screen.refresh()
        self.display_screen.instruction = None

        info_rect = pg.Rect(0.3 * self.display_size.x, 0, 0.7 * self.display_size.x, 0.8 * self.display_size.y)
        pg.draw.rect(self.display_screen.surface, Colours.white.value, info_rect)
        self.display_screen.add_multiline_text("Trail Making Test", rect=info_rect.scale_by(0.9, 0.9), font_size=50)

        # Exactly the same button as Shapes
        button_rect = pg.Rect((self.display_size - pg.Vector2(300, 200)) / 2, (300, 200))
        start_button = GameButton(position=button_rect.topleft, size=button_rect.size, text="START", id=1)
        self.touch_screen.sprites = GameObjects([start_button])

        self.top_screen.blit(self.display_screen.get_surface(), (0, 0))
        self.bottom_screen.blit(self.touch_screen.get_surface(), (0, 0))
        pg.display.flip()

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

    def entry_sequence(self):
        self.running = True
        self._instruction_loop()
        self._update_display()

    def exit_sequence(self):
        """Show completion screen briefly then hand back to orchestrator."""
        if self.results:
            self._update_display()
            pg.time.wait(2500)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def loop(self):
        """Main game loop — called by the orchestrator."""
        self.entry_sequence()
        clock = pg.time.Clock()

        while self.running:
            clock.tick(FPS)

            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.running = False
                    elif event.key == pg.K_TAB and not self.completed:
                        # Toggle input mode
                        self.input_mode = (
                            MODE_MOUSE if self.input_mode == MODE_STYLUS else MODE_STYLUS
                        )

                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    # Convert to bottom-screen coordinates
                    raw = pg.Vector2(pg.mouse.get_pos())
                    pos = (int(raw.x), int(raw.y - self.display_size.y))

                    if self.input_mode == MODE_MOUSE:
                        self._handle_click(pos)
                    else:
                        self.drawing = True
                        self.draw_path = [pos]
                        if self.start_time:
                            self._handle_touch_movement(pos, time.time())

                elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                    self.drawing = False

                elif event.type == pg.MOUSEMOTION:
                    if self.drawing and not self.completed and self.input_mode == MODE_STYLUS:
                        raw = pg.Vector2(pg.mouse.get_pos())
                        pos = (int(raw.x), int(raw.y - self.display_size.y))
                        self.draw_path.append(pos)
                        self._handle_touch_movement(pos, time.time())

            # Auto-advance after completion
            if self.completed:
                self._update_display()
                pg.time.wait(2500)
                self.running = False
                break

            self._update_display()

        self.exit_sequence()
        
