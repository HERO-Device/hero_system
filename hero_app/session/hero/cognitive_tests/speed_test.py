"""
Speed Test Module - Reaction time assessment

Measures patient's reaction time by clicking randomly appearing circles.
"""

import math
import random
import time
import pygame as pg

# Try to import from consultation structure
try:
    from hero.consultation.screen import Colours

    INTEGRATED = True
except ImportError:
    # Standalone mode
    from enum import Enum


    class Colours(Enum):
        white = pg.Color(255, 255, 255)
        black = pg.Color(0, 0, 0)
        red = pg.Color(181, 67, 67)
        blue = pg.Color(67, 113, 181)


    INTEGRATED = False


class SpeedTest:
    """Reaction time test - click circles as fast as possible."""

    def __init__(self, parent=None, auto_run=False, num_trials=10):
        self.parent = parent
        self.auto_run = auto_run
        self.num_trials = num_trials
        self.running = False

        # Results
        self.reaction_times = []
        self.click_count = 0
        self.average_time = 0

        # Visual settings
        self.colors = [
            pg.Color(181, 67, 67),  # Red
            pg.Color(120, 0, 120),  # Purple
            pg.Color(67, 113, 181),  # Blue
            pg.Color(255, 102, 0),  # Orange
            pg.Color(255, 0, 255),  # Pink
            pg.Color(153, 204, 0),  # Green
        ]

        # Circle state
        self.circle_x = 0
        self.circle_y = 0
        self.circle_radius = 0
        self.circle_color = None
        self.start_time = None

    def setup_new_circle(self):
        """Generate new random circle."""
        if self.parent:
            width = int(self.parent.bottom_screen.get_width())
            height = int(self.parent.bottom_screen.get_height())
        else:
            width, height = 600, 600

        self.circle_radius = random.randint(15, 30)
        self.circle_x = random.randint(self.circle_radius + 10, width - self.circle_radius - 10)
        self.circle_y = random.randint(self.circle_radius + 40, height - self.circle_radius - 100)
        self.circle_color = random.choice(self.colors)
        self.start_time = time.time()

    def draw_circle(self, surface):
        """Draw circle on surface."""
        if self.circle_radius > 0:
            pg.draw.circle(surface, self.circle_color,
                           (int(self.circle_x), int(self.circle_y)),
                           self.circle_radius)

    def check_click(self, pos):
        """Check if click hits circle."""
        x, y = pos
        distance_sq = (x - self.circle_x) ** 2 + (y - self.circle_y) ** 2
        return distance_sq < self.circle_radius ** 2

    def handle_successful_click(self):
        """Process successful click."""
        reaction_time = time.time() - self.start_time
        self.reaction_times.append(reaction_time)
        self.click_count += 1

        print(f"Click {self.click_count}/{self.num_trials}: {reaction_time:.3f}s")

        if self.click_count >= self.num_trials:
            self.calculate_results()
            self.running = False
        else:
            self.setup_new_circle()

    def calculate_results(self):
        """Calculate average reaction time."""
        if self.reaction_times:
            self.average_time = sum(self.reaction_times) / len(self.reaction_times)
            print(f"\n✓ Average: {self.average_time:.3f}s")

    def loop(self):
        """Main test loop."""
        self.running = True
        self.click_count = 0
        self.reaction_times = []

        if self.parent:
            self.run_integrated()
        else:
            self.run_standalone()

        return {
            "reaction_times": self.reaction_times,
            "average_time": round(self.average_time, 3),
            "click_count": self.click_count,
            "completed": self.click_count >= self.num_trials
        }

    def run_integrated(self):
        """Run with parent Consultation."""
        self.parent.display_screen.instruction = "Click circles fast!"
        self.parent.update_display()

        if self.parent.config.speech and not self.auto_run:
            self.parent.speak_text("Click the circles as quickly as you can")

        self.setup_new_circle()

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = self.parent.get_relative_mose_pos()
                    if self.check_click(mouse_pos):
                        self.handle_successful_click()
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    self.running = False

            # Draw
            white = Colours.white.value if INTEGRATED else Colours.white
            self.parent.touch_screen.surface.fill(white)
            self.draw_circle(self.parent.touch_screen.surface)
            self.parent.update_display()

    def run_standalone(self):
        """Run standalone for testing."""
        pg.init()
        screen = pg.display.set_mode((600, 600))
        pg.display.set_caption("Speed Test")
        clock = pg.time.Clock()
        font = pg.font.Font(None, 36)

        self.setup_new_circle()

        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if self.check_click(pg.mouse.get_pos()):
                        self.handle_successful_click()
                elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                    self.running = False

            # Draw
            screen.fill((255, 255, 255))
            self.draw_circle(screen)

            title = font.render("Speed Test", True, (255, 0, 0))
            screen.blit(title, (200, 20))

            counter = font.render(f"Click {self.click_count}/{self.num_trials}", True, (0, 0, 0))
            screen.blit(counter, (20, 550))

            pg.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    print("=" * 50)
    print("Speed Test - Standalone")
    print("=" * 50)

    pg.init()
    test = SpeedTest(num_trials=5)
    results = test.loop()

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Average: {results['average_time']:.3f}s")
    print(f"Times: {[f'{t:.3f}s' for t in results['reaction_times']]}")
    pg.quit()
