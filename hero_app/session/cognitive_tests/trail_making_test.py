"""
Trail Making Test (TMT) — medical assessment game.

Patient clicks numbered circles in order (1 → 2 → 3 ...).
Tracks completion time, errors, path efficiency, smoothness, and pauses.

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


class TMTCircle:
    """
    A numbered target circle for the Trail Making Test.

    Attributes:
        x: Pixel x-coordinate of the circle centre.
        y: Pixel y-coordinate of the circle centre.
        label: Integer label displayed inside the circle.
        radius: Circle radius in pixels.
        completed: True once the patient has hit this circle.
        entry_time: Elapsed time in seconds when the circle was hit.
        entry_position: (x, y) pixel position of the hit event.
    """

    def __init__(self, x, y, label, radius):
        """
        Initialise a TMT circle.

        Args:
            x: Pixel x-coordinate of the circle centre.
            y: Pixel y-coordinate of the circle centre.
            label: Integer label to display inside the circle.
            radius: Circle radius in pixels.
        """
        self.x = x
        self.y = y
        self.label = label
        self.radius = radius
        self.completed = False
        self.entry_time = None
        self.entry_position = None

    def draw(self, surface, is_current=False):
        """
        Render the circle onto a surface.

        Args:
            surface: pygame Surface to draw onto.
            is_current: If True, highlight the circle as the next target.
        """
        color = GREEN if self.completed else (LIGHT_BLUE if is_current else WHITE)
        pg.draw.circle(surface, color, (self.x, self.y), self.radius)
        pg.draw.circle(surface, BLACK, (self.x, self.y), self.radius, 2)
        font = pg.font.Font(None, 36)
        text = font.render(str(self.label), True, BLACK)
        surface.blit(text, text.get_rect(center=(self.x, self.y)))

    def contains_point(self, pos):
        """
        Test whether a position falls within the circle.

        Args:
            pos: (x, y) position to test.

        Returns:
            True if pos is within the circle radius, False otherwise.
        """
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        return math.sqrt(dx * dx + dy * dy) <= self.radius


class PerformanceMetrics:
    """
    Accumulates and computes kinematic metrics for a TMT run.

    Tracks path points, movement speed, pauses, and provides summary
    statistics including path efficiency and smoothness.

    Attributes:
        path_points: List of dicts with 'position' and 'timestamp' keys.
        optimal_distance: Straight-line sum between consecutive circles.
        actual_distance: Total length of the recorded path.
        pauses: List of pause event dicts with 'start' and 'duration'.
        current_pause_start: Timestamp of the current pause start, or None.
        last_movement_time: Timestamp of the last detected movement.
        movement_speeds: List of instantaneous speed samples in px/s.
    """

    def __init__(self):
        """Initialise all metric accumulators."""
        self.reset()

    def reset(self):
        """Reset all accumulators to their initial empty state."""
        self.path_points = []
        self.optimal_distance = 0
        self.actual_distance = 0
        self.pauses = []
        self.current_pause_start = None
        self.last_movement_time = None
        self.movement_speeds = []

    def add_path_point(self, pos, timestamp):
        """
        Record a path point and update speed and pause tracking.

        Args:
            pos: (x, y) position in bottom-screen pixel coordinates.
            timestamp: Elapsed time in seconds since test start.
        """
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
        """
        Compute total path length from all recorded path points.

        Returns:
            Total Euclidean path length in pixels.
        """
        total = 0
        for i in range(1, len(self.path_points)):
            prev = self.path_points[i - 1]['position']
            curr = self.path_points[i]['position']
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def calculate_optimal_distance(self, circles):
        """
        Compute the shortest possible path through all circles in order.

        Args:
            circles: List of TMTCircle objects in completion order.

        Returns:
            Sum of straight-line distances between consecutive circle centres.
        """
        total = 0
        for i in range(1, len(circles)):
            dx = circles[i].x - circles[i - 1].x
            dy = circles[i].y - circles[i - 1].y
            total += math.sqrt(dx * dx + dy * dy)
        return total

    def calculate_path_efficiency(self):
        """
        Compute path efficiency as a percentage of optimal over actual distance.

        Returns:
            Float in [0, 100]; 100 means the patient took the optimal path.
        """
        if self.actual_distance == 0:
            return 100.0
        return min(100.0, (self.optimal_distance / self.actual_distance) * 100)

    def calculate_path_smoothness(self):
        """
        Compute path smoothness from direction change variability.

        Analyses direction changes over a sliding window and returns a score
        where 100 indicates perfectly straight segments.

        Returns:
            Float in [0, 100]; higher is smoother.
        """
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
        """
        Compile all metrics into a results summary dict.

        Args:
            completion_time: Total elapsed time for the test in seconds.
            completed_circles: List of TMTCircle objects that were completed.

        Returns:
            Dict with keys: completion_time, path_efficiency, path_smoothness,
            total_distance, optimal_distance, pause_count, total_pause_time,
            average_speed, speed_variability.
        """
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
        """
        Compute the population standard deviation of a list of values.

        Args:
            values: List of numeric values.

        Returns:
            Standard deviation, or 0 if fewer than 2 values.
        """
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


class TrailMakingTest:
    """
    Trail Making Test game for the HERO consultation system.

    Generates a random layout of numbered circles on the touchscreen and
    records the patient's path through them in order. Supports both click
    (mouse) and continuous stylus (touchscreen) input modes.

    Attributes:
        parent: Parent Consultation instance, or None in standalone mode.
        auto_run: If True, skip instruction screens and simulate input.
        running: Main loop control flag.
        difficulty: Circle size setting — 'small', 'medium', or 'large'.
        test_type: TMT variant label ('A' for numbers only).
        circle_radius: Pixel radius for each circle.
        display_size: Vector2 dimensions of the display area.
        display_screen: DisplayScreen for the upper screen.
        touch_screen: TouchScreen for the lower screen.
        circles: List of TMTCircle objects for this run.
        current_index: Index of the next circle to be hit.
        path: List of dicts recording each successfully hit circle.
        errors: Count of incorrect circle hits.
        start_time: Unix timestamp of the first circle hit.
        end_time: Unix timestamp of test completion.
        completed: True once all circles have been hit.
        drawing: True while the stylus is held down in touchscreen mode.
        draw_path: List of (x, y) positions forming the current stylus stroke.
        input_mode: Current mode string — MODE_STYLUS or MODE_MOUSE.
        metrics: PerformanceMetrics instance accumulating kinematic data.
        results: Dict populated by _complete_test() with summary statistics.
    """

    def __init__(self, parent=None, auto_run=False,
                 difficulty='medium', test_type='A'):
        """
        Initialise the Trail Making Test and generate the circle layout.

        Args:
            parent: Parent Consultation instance. If provided, screens are shared.
            auto_run: If True, skip instruction screens and simulate input.
            difficulty: Circle size — 'small', 'medium', or 'large'.
            test_type: TMT variant label. Currently only 'A' (numeric) is used.
        """
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

    def _generate_circles(self):
        """Generate non-overlapping numbered circles placed randomly on the bottom screen."""
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

    def _handle_click(self, pos):
        """
        Register a click event in mouse input mode.

        Records a hit if the click lands on the expected next circle, or
        increments the error count if a different circle is clicked.

        Args:
            pos: (x, y) position in bottom-screen pixel coordinates.
        """
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
        """
        Handle continuous stylus movement in touchscreen input mode.

        Passes every position to the metrics tracker and auto-completes a
        circle when the stylus enters its boundary.

        Args:
            pos: (x, y) position in bottom-screen pixel coordinates.
            timestamp: Absolute time.time() value for the event.
        """
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
        """Mark the test complete, compile summary metrics, and populate self.results."""
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

    def _draw(self, surface):
        """
        Draw the full game state onto the given surface.

        Args:
            surface: pygame Surface to render onto (sized to display_size).
        """
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

        # Completion overlay
        if self.completed and self.results:
            overlay = pg.Surface((w, h), pg.SRCALPHA)
            overlay.fill((255, 255, 255, 220))
            surface.blit(overlay, (0, 0))

    def _update_display(self):
        """Render the current game state to both physical screens."""
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

    def _instruction_loop(self):
        """
        Show the instruction screen and wait for the patient to press Start.

        No-ops in auto_run mode.
        """
        if self.auto_run:
            return

        self.display_screen.state = 1
        self.display_screen.refresh()
        self.display_screen.instruction = None

        info_rect = pg.Rect(0.3 * self.display_size.x, 0, 0.7 * self.display_size.x, 0.8 * self.display_size.y)
        pg.draw.rect(self.display_screen.surface, Colours.white.value, info_rect)
        self.display_screen.add_multiline_text("Trail Making Test", rect=info_rect.scale_by(0.9, 0.9), font_size=50)

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
        """Show instructions and begin the game."""
        self.running = True
        self._instruction_loop()
        self._update_display()

    def exit_sequence(self):
        """
        Display the completion screen briefly before returning control.

        Shows the final state for 2.5 seconds if results are available.
        """
        if self.results:
            self._update_display()
            pg.time.wait(2500)

    def loop(self):
        """
        Main game loop — called by the orchestrator.

        Processes input events, updates game state, and renders each frame
        until all circles are completed or the loop is exited.
        """
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
