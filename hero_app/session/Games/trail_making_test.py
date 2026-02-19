"""
Trail Making Test (TMT) - Medical Assessment Game
Enhanced version with advanced performance metrics for medical device application

Features:
- Continuous touch detection for touchscreen mode
- Path efficiency scoring
- Hesitation/pause detection
- Path smoothness analysis (motor control assessment)
- Difficulty levels (small/medium/large circles)
- Comprehensive anonymised data export

Input Modes:
- TOUCHSCREEN: Continuous drawing with stylus (auto-detects circle entry)
- MOUSE: Click-based selection for laptop testing
"""

import pygame
import random
import math
import time
import json
from datetime import datetime

# Initialize Pygame
pygame.init()

# Configuration
WIDTH, HEIGHT = 800, 600
NUM_CIRCLES_A = 10  # Using 10 for demo (standard TMT Part A uses 25)
FPS = 60

# Difficulty settings
DIFFICULTY_SETTINGS = {
    'small': {'radius': 20, 'label': 'SMALL (Hard)'},
    'medium': {'radius': 30, 'label': 'MEDIUM (Standard)'},
    'large': {'radius': 45, 'label': 'LARGE (Easy)'}
}

# Performance thresholds
PAUSE_THRESHOLD = 1.5  # seconds - movement pause detection
MIN_MOVEMENT_SPEED = 5  # pixels/frame - below this is considered paused
SMOOTHNESS_WINDOW = 10  # points to analyze for smoothness

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLUE = (0, 100, 255)
LIGHT_BLUE = (150, 200, 255)
ORANGE = (255, 165, 0)
YELLOW = (255, 220, 0)

# Input modes
MODE_STYLUS = 'touchscreen'
MODE_MOUSE = 'mouse'

class Circle:
    def __init__(self, x, y, label, radius):
        self.x = x
        self.y = y
        self.label = label
        self.radius = radius
        self.completed = False
        self.entry_time = None
        self.entry_position = None

    def draw(self, screen, is_current=False):
        color = GREEN if self.completed else (LIGHT_BLUE if is_current else WHITE)
        pygame.draw.circle(screen, color, (self.x, self.y), self.radius)
        pygame.draw.circle(screen, BLACK, (self.x, self.y), self.radius, 2)

        font = pygame.font.Font(None, 36)
        text = font.render(str(self.label), True, BLACK)
        text_rect = text.get_rect(center=(self.x, self.y))
        screen.blit(text, text_rect)

    def contains_point(self, pos):
        dx = pos[0] - self.x
        dy = pos[1] - self.y
        return math.sqrt(dx*dx + dy*dy) <= self.radius

class PerformanceMetrics:
    """Advanced performance metrics calculator"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.path_points = []
        self.circle_times = []
        self.optimal_distance = 0
        self.actual_distance = 0
        self.pauses = []
        self.current_pause_start = None
        self.last_movement_time = None
        self.movement_speeds = []

    def add_path_point(self, pos, timestamp):
        """Add a point to the drawing path"""
        self.path_points.append({
            'position': pos,
            'timestamp': timestamp
        })

        # Calculate movement speed if we have previous point
        if len(self.path_points) > 1:
            prev = self.path_points[-2]
            dx = pos[0] - prev['position'][0]
            dy = pos[1] - prev['position'][1]
            distance = math.sqrt(dx*dx + dy*dy)
            time_delta = timestamp - prev['timestamp']

            if time_delta > 0:
                speed = distance / time_delta
                self.movement_speeds.append(speed)

                # Detect pauses (low speed for extended time)
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
        """Calculate total distance traveled along the path"""
        total = 0
        for i in range(1, len(self.path_points)):
            prev = self.path_points[i-1]['position']
            curr = self.path_points[i]['position']
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            total += math.sqrt(dx*dx + dy*dy)
        return total

    def calculate_optimal_distance(self, circles):
        """Calculate optimal straight-line distance between circles"""
        total = 0
        for i in range(1, len(circles)):
            prev = circles[i-1]
            curr = circles[i]
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            total += math.sqrt(dx*dx + dy*dy)
        return total

    def calculate_path_efficiency(self):
        """Calculate path efficiency score (0-100%)"""
        if self.actual_distance == 0:
            return 100.0
        efficiency = (self.optimal_distance / self.actual_distance) * 100
        return min(100.0, efficiency)

    def calculate_path_smoothness(self):
        """Calculate path smoothness using direction change analysis"""
        if len(self.path_points) < SMOOTHNESS_WINDOW:
            return 100.0  # Not enough data

        # Calculate direction changes
        direction_changes = []
        for i in range(SMOOTHNESS_WINDOW, len(self.path_points)):
            points = self.path_points[i-SMOOTHNESS_WINDOW:i]

            # Calculate angles between consecutive segments
            angles = []
            for j in range(1, len(points)-1):
                p1 = points[j-1]['position']
                p2 = points[j]['position']
                p3 = points[j+1]['position']

                # Vector from p1 to p2
                v1x = p2[0] - p1[0]
                v1y = p2[1] - p1[1]

                # Vector from p2 to p3
                v2x = p3[0] - p2[0]
                v2y = p3[1] - p2[1]

                # Calculate angle between vectors
                mag1 = math.sqrt(v1x*v1x + v1y*v1y)
                mag2 = math.sqrt(v2x*v2x + v2y*v2y)

                if mag1 > 0 and mag2 > 0:
                    dot = v1x*v2x + v1y*v2y
                    cos_angle = dot / (mag1 * mag2)
                    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp
                    angle = math.acos(cos_angle)
                    angles.append(abs(angle))

            if angles:
                avg_change = sum(angles) / len(angles)
                direction_changes.append(avg_change)

        if not direction_changes:
            return 100.0

        # Lower average change = smoother path
        # Convert to 0-100 scale (0 radians = 100%, π radians = 0%)
        avg_change = sum(direction_changes) / len(direction_changes)
        smoothness = 100 * (1 - (avg_change / math.pi))
        return max(0.0, min(100.0, smoothness))

    def get_summary(self, completion_time, completed_circles):
        """Get comprehensive metrics summary"""
        self.actual_distance = self.calculate_actual_distance()
        self.optimal_distance = self.calculate_optimal_distance(completed_circles)

        return {
            'completion_time': completion_time,
            'path_efficiency': round(self.calculate_path_efficiency(), 2),
            'path_smoothness': round(self.calculate_path_smoothness(), 2),
            'total_distance': round(self.actual_distance, 2),
            'optimal_distance': round(self.optimal_distance, 2),
            'pause_count': len(self.pauses),
            'total_pause_time': round(sum(p['duration'] for p in self.pauses), 2),
            'pauses': self.pauses,
            'average_speed': round(sum(self.movement_speeds) / len(self.movement_speeds), 2) if self.movement_speeds else 0,
            'speed_variability': round(self._calculate_std(self.movement_speeds), 2) if len(self.movement_speeds) > 1 else 0
        }

    def _calculate_std(self, values):
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

class TrailMakingTest:
    def __init__(self, test_type='A', difficulty='medium'):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Trail Making Test - Part ' + test_type)
        self.clock = pygame.time.Clock()
        self.test_type = test_type
        self.difficulty = difficulty
        self.circle_radius = DIFFICULTY_SETTINGS[difficulty]['radius']

        self.circles = []
        self.current_index = 0
        self.path = []
        self.errors = 0
        self.start_time = None
        self.end_time = None
        self.completed = False
        self.drawing = False
        self.draw_path = []

        # Input mode - default to mouse for laptop testing
        self.input_mode = MODE_MOUSE

        # Performance metrics
        self.metrics = PerformanceMetrics()

        self._generate_circles()

    def _generate_circles(self):
        """Generate random circle positions ensuring no overlap"""
        labels = list(range(1, NUM_CIRCLES_A + 1))

        margin = self.circle_radius + 50
        attempts = 0
        max_attempts = 1000

        while len(self.circles) < NUM_CIRCLES_A and attempts < max_attempts:
            x = random.randint(margin, WIDTH - margin)
            y = random.randint(margin + 100, HEIGHT - margin - 50)

            # Check for overlap with existing circles
            valid = True
            for circle in self.circles:
                dist = math.sqrt((x - circle.x)**2 + (y - circle.y)**2)
                if dist < (self.circle_radius + circle.radius) * 2:
                    valid = False
                    break

            if valid:
                self.circles.append(Circle(x, y, labels[len(self.circles)], self.circle_radius))

            attempts += 1

    def toggle_input_mode(self):
        """Toggle between touchscreen/stylus and mouse input modes"""
        if self.input_mode == MODE_STYLUS:
            self.input_mode = MODE_MOUSE
        else:
            self.input_mode = MODE_STYLUS
        print(f"Input mode switched to: {self.input_mode.upper()}")

    def check_circle_collision(self, pos):
        """Check if current position is inside the expected circle (touchscreen mode)"""
        if self.completed or self.current_index >= len(self.circles):
            return False

        expected_circle = self.circles[self.current_index]
        return expected_circle.contains_point(pos)

    def handle_touch_movement(self, pos, timestamp):
        """Handle continuous touch movement (touchscreen mode)"""
        if self.completed:
            return

        if self.start_time is None:
            self.start_time = time.time()

        # Add to path tracking
        self.metrics.add_path_point(pos, timestamp - self.start_time)

        # Check if we've entered the next circle
        if self.check_circle_collision(pos):
            expected_circle = self.circles[self.current_index]

            if not expected_circle.completed:
                expected_circle.completed = True
                expected_circle.entry_time = timestamp - self.start_time
                expected_circle.entry_position = pos

                self.path.append({
                    'circle': expected_circle.label,
                    'timestamp': expected_circle.entry_time,
                    'position': pos,
                    'input_method': self.input_mode
                })

                self.current_index += 1

                if self.current_index >= len(self.circles):
                    self._complete_test()

    def handle_click(self, pos):
        """Handle circle selection via click (mouse mode)"""
        if self.completed or self.input_mode != MODE_MOUSE:
            return

        if self.start_time is None:
            self.start_time = time.time()

        timestamp = time.time() - self.start_time
        expected_circle = self.circles[self.current_index]

        # Check if clicked on the correct circle
        if expected_circle.contains_point(pos):
            expected_circle.completed = True
            expected_circle.entry_time = timestamp
            expected_circle.entry_position = pos

            self.path.append({
                'circle': expected_circle.label,
                'timestamp': timestamp,
                'position': pos,
                'input_method': self.input_mode
            })

            # Add metrics for mouse mode (straight line between clicks)
            if self.current_index > 0:
                prev_circle = self.circles[self.current_index - 1]
                self.metrics.add_path_point(prev_circle.entry_position, prev_circle.entry_time)
            self.metrics.add_path_point(pos, timestamp)

            self.current_index += 1

            if self.current_index >= len(self.circles):
                self._complete_test()
        else:
            # Check if clicked on wrong circle
            for circle in self.circles:
                if circle.contains_point(pos) and not circle.completed:
                    self.errors += 1
                    break

    def _complete_test(self):
        """Complete the test and save results"""
        self.completed = True
        self.end_time = time.time()
        completion_time = self.end_time - self.start_time

        # Get comprehensive performance metrics
        completed_circles = [c for c in self.circles if c.completed]
        metrics_summary = self.metrics.get_summary(completion_time, completed_circles)

        # Prepare anonymised data for storage
        game_data = {
            'test_type': self.test_type,
            'difficulty': self.difficulty,
            'circle_radius': self.circle_radius,
            'completion_time_seconds': round(completion_time, 2),
            'errors': self.errors,
            'num_circles': NUM_CIRCLES_A,
            'timestamp': datetime.now().isoformat(),
            'input_mode': self.input_mode,
            'path_data': self.path,
            'performance_metrics': metrics_summary
        }

        # Save to JSON file (anonymised)
        filename = f'tmt_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(filename, 'w') as f:
            json.dump(game_data, f, indent=2)

        print(f"{'='*60}")
        print(f"TEST COMPLETED!")
        print(f"{'='*60}")
        print(f"Time: {completion_time:.2f}s | Errors: {self.errors}")
        print(f"Path Efficiency: {metrics_summary['path_efficiency']:.1f}%")
        print(f"Path Smoothness: {metrics_summary['path_smoothness']:.1f}%")
        print(f"Pauses: {metrics_summary['pause_count']} ({metrics_summary['total_pause_time']:.1f}s total)")
        print(f"Distance: {metrics_summary['total_distance']:.0f}px (optimal: {metrics_summary['optimal_distance']:.0f}px)")
        print(f"Data saved to: {filename}")
        print(f"{'='*60}")

    def draw_connecting_line(self, screen):
        """Draw line from last completed circle to mouse position"""
        if self.current_index > 0 and not self.completed:
            last_circle = self.circles[self.current_index - 1]
            mouse_pos = pygame.mouse.get_pos()
            pygame.draw.line(screen, GRAY, (last_circle.x, last_circle.y), mouse_pos, 2)

    def draw_completed_lines(self, screen):
        """Draw lines between completed circles"""
        for i in range(1, self.current_index):
            prev_circle = self.circles[i - 1]
            curr_circle = self.circles[i]
            pygame.draw.line(screen, BLUE, (prev_circle.x, prev_circle.y), 
                           (curr_circle.x, curr_circle.y), 3)

    def draw_stylus_path(self, screen):
        """Draw the path traced by the stylus (touchscreen mode only)"""
        if len(self.draw_path) > 1 and self.input_mode == MODE_STYLUS:
            pygame.draw.lines(screen, RED, False, self.draw_path, 2)

    def draw_mode_indicator(self, screen):
        """Draw input mode indicator"""
        font = pygame.font.Font(None, 20)
        mode_display = "TOUCHSCREEN" if self.input_mode == MODE_STYLUS else "MOUSE"
        mode_text = f"Mode: {mode_display}"
        color = ORANGE if self.input_mode == MODE_STYLUS else BLUE

        # Background box
        text_surface = font.render(mode_text, True, WHITE)
        padding = 8
        box_width = text_surface.get_width() + padding * 2
        box_height = text_surface.get_height() + padding
        box_rect = pygame.Rect(WIDTH - box_width - 10, 10, box_width, box_height)
        pygame.draw.rect(screen, color, box_rect)
        pygame.draw.rect(screen, BLACK, box_rect, 2)

        text_rect = text_surface.get_rect(center=box_rect.center)
        screen.blit(text_surface, text_rect)

        # Difficulty indicator
        diff_text = DIFFICULTY_SETTINGS[self.difficulty]['label']
        diff_surface = font.render(diff_text, True, BLACK)
        screen.blit(diff_surface, (WIDTH - diff_surface.get_width() - 10, box_rect.bottom + 5))

    def draw_live_metrics(self, screen):
        """Draw live performance metrics during test"""
        if not self.start_time or self.completed:
            return

        elapsed = time.time() - self.start_time
        font = pygame.font.Font(None, 22)
        y_offset = HEIGHT - 80

        # Current metrics
        efficiency = self.metrics.calculate_path_efficiency()
        smoothness = self.metrics.calculate_path_smoothness()
        pauses = len(self.metrics.pauses)

        metrics_text = [
            f"Efficiency: {efficiency:.0f}%",
            f"Smoothness: {smoothness:.0f}%",
            f"Pauses: {pauses}"
        ]

        for i, text in enumerate(metrics_text):
            surface = font.render(text, True, DARK_GRAY)
            screen.blit(surface, (10, y_offset + i * 25))

    def draw(self):
        """Main draw method"""
        self.screen.fill(WHITE)

        # Draw title and instructions
        font = pygame.font.Font(None, 28)
        title = font.render(f'Trail Making Test - Part {self.test_type}', True, BLACK)
        self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 15))

        if not self.completed:
            inst_font = pygame.font.Font(None, 20)
            if self.input_mode == MODE_STYLUS:
                instruction = inst_font.render('Draw continuously through circles: 1 → 2 → 3 ...', True, BLACK)
            else:
                instruction = inst_font.render('Click circles in order: 1 → 2 → 3 ...', True, BLACK)
            self.screen.blit(instruction, (WIDTH//2 - instruction.get_width()//2, 45))

        # Draw completed connecting lines
        self.draw_completed_lines(self.screen)

        # Draw stylus path
        self.draw_stylus_path(self.screen)

        # Draw line to mouse cursor
        self.draw_connecting_line(self.screen)

        # Draw all circles
        for i, circle in enumerate(self.circles):
            is_current = (i == self.current_index)
            circle.draw(self.screen, is_current)

        # Draw stats
        stats_font = pygame.font.Font(None, 26)
        if self.start_time and not self.completed:
            elapsed = time.time() - self.start_time
            time_text = stats_font.render(f'Time: {elapsed:.1f}s', True, BLACK)
            self.screen.blit(time_text, (10, 10))

        error_color = RED if self.errors > 0 else BLACK
        error_text = stats_font.render(f'Errors: {self.errors}', True, error_color)
        self.screen.blit(error_text, (10, 35))

        # Draw live metrics
        self.draw_live_metrics(self.screen)

        # Draw input mode indicator
        self.draw_mode_indicator(self.screen)

        # Draw completion message
        if self.completed:
            metrics_summary = self.metrics.get_summary(self.end_time - self.start_time, 
                                                       [c for c in self.circles if c.completed])

            # Semi-transparent overlay
            overlay = pygame.Surface((WIDTH, HEIGHT))
            overlay.set_alpha(220)
            overlay.fill(WHITE)
            self.screen.blit(overlay, (0, 0))

            # Completion message
            complete_font = pygame.font.Font(None, 48)
            msg = complete_font.render('Test Completed!', True, GREEN)
            self.screen.blit(msg, (WIDTH//2 - msg.get_width()//2, 100))

            # Stats
            stats_font_large = pygame.font.Font(None, 28)
            stats_y = 170

            stats_lines = [
                f"Time: {metrics_summary['completion_time']:.2f}s",
                f"Errors: {self.errors}",
                f"Path Efficiency: {metrics_summary['path_efficiency']:.1f}%",
                f"Path Smoothness: {metrics_summary['path_smoothness']:.1f}%",
                f"Pauses: {metrics_summary['pause_count']} ({metrics_summary['total_pause_time']:.1f}s)",
                f"Distance: {metrics_summary['total_distance']:.0f}px (optimal: {metrics_summary['optimal_distance']:.0f}px)"
            ]

            for i, line in enumerate(stats_lines):
                text = stats_font_large.render(line, True, BLACK)
                self.screen.blit(text, (WIDTH//2 - text.get_width()//2, stats_y + i * 35))

            # Instructions
            inst_font = pygame.font.Font(None, 24)
            restart_msg = inst_font.render('R: Restart | Q: Quit | TAB: Change Mode', True, DARK_GRAY)
            self.screen.blit(restart_msg, (WIDTH//2 - restart_msg.get_width()//2, HEIGHT - 60))

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        running = True

        print(f"{'='*60}")
        print(f"TRAIL MAKING TEST - Part {self.test_type}")
        print(f"{'='*60}")
        print(f"Difficulty: {DIFFICULTY_SETTINGS[self.difficulty]['label']}")
        print(f"Circle radius: {self.circle_radius}px")
        print(f"Input mode: {self.input_mode.upper()}")
        print(f"Controls:")
        if self.input_mode == MODE_STYLUS:
            print(f"  - Draw continuously through circles in order")
        else:
            print(f"  - Click circles in numerical order")
        print(f"  - TAB: Toggle input mode")
        print(f"  - R: Restart | Q: Quit")
        print(f"{'='*60}")

        while running:
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        # Mouse mode: register click
                        if self.input_mode == MODE_MOUSE:
                            self.handle_click(event.pos)
                        # Touchscreen mode: start drawing
                        else:
                            self.drawing = True
                            self.draw_path = [event.pos]
                            if self.start_time:
                                self.handle_touch_movement(event.pos, time.time())

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.drawing = False

                elif event.type == pygame.MOUSEMOTION:
                    # Touchscreen mode: continuous drawing and collision detection
                    if self.drawing and not self.completed and self.input_mode == MODE_STYLUS:
                        self.draw_path.append(event.pos)
                        self.handle_touch_movement(event.pos, time.time())

                elif event.type == pygame.KEYDOWN:
                    # Toggle input mode
                    if event.key == pygame.K_TAB and not self.completed:
                        self.toggle_input_mode()

                    # Restart or quit
                    elif self.completed or self.current_index == 0:
                        if event.key == pygame.K_r:
                            self.__init__(self.test_type, self.difficulty)
                        elif event.key == pygame.K_q:
                            running = False
                        # Change difficulty
                        elif event.key == pygame.K_1:
                            self.__init__(self.test_type, 'small')
                        elif event.key == pygame.K_2:
                            self.__init__(self.test_type, 'medium')
                        elif event.key == pygame.K_3:
                            self.__init__(self.test_type, 'large')

            self.draw()

        pygame.quit()

if __name__ == '__main__':
    print("Difficulty Selection:")
    print("1. Small circles (Hard)")
    print("2. Medium circles (Standard)")
    print("3. Large circles (Easy)")
    choice = input("Select difficulty (1/2/3) [default: 2]: ").strip()

    difficulty_map = {'1': 'small', '2': 'medium', '3': 'large'}
    difficulty = difficulty_map.get(choice, 'medium')

    # Create and run Trail Making Test Part A
    game = TrailMakingTest(test_type='A', difficulty=difficulty)
    game.run()