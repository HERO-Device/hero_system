"""
Binding Test Module - Feature binding memory assessment

Tests ability to bind features together in memory.
Shows two images sequentially and asks if they're the same or different.
Similar to memory test but focuses on binding visual features.
"""

import os
import time
import pygame as pg
import pandas as pd

# Try to import from hero structure
try:
    from hero.consultation.screen import Colours, BlitLocation
    INTEGRATED = True
except ImportError:
    from enum import Enum
    class Colours(Enum):
        white = pg.Color(255, 255, 255)
        black = pg.Color(0, 0, 0)
        grey = pg.Color(200, 200, 200)

    class BlitLocation(Enum):
        centre = 8

    INTEGRATED = False


class BindingTest:
    """
    Feature binding test - testing visual feature memory binding.

    Shows image 1 for 2 seconds, blank screen for 1 second,
    then image 2. Patient indicates if images are same or different.

    This tests the ability to bind visual features (color, shape, position)
    together in memory - important for detecting cognitive decline.
    """

    def __init__(self, parent=None, auto_run=False, num_trials=32,
                 image_folder="games/Images", answers_file="games/bindinganswers.xlsx"):
        """
        Initialize Binding Test.

        Args:
            parent: Parent Consultation object
            auto_run: If True, runs automatically
            num_trials: Number of image pairs to show (default: 32)
            image_folder: Path to folder containing images
            answers_file: Path to Excel file with correct answers
        """
        self.parent = parent
        self.auto_run = auto_run
        self.num_trials = num_trials
        self.running = False

        # File paths
        self.image_folder = image_folder
        self.answers_file = answers_file

        # Results storage
        self.responses = []
        self.correct_answers = []
        self.is_correct = []
        self.response_times = []

        # Load answer key
        self.answer_df = None
        if os.path.exists(answers_file):
            try:
                self.answer_df = pd.read_excel(answers_file)
                print(f"✓ Loaded answers from {answers_file}")
            except Exception as e:
                print(f"⚠ Warning: Could not load answers file: {e}")
        else:
            print(f"⚠ Warning: Answers file not found: {answers_file}")

        # Timing settings
        self.image_display_time = 2000  # ms
        self.pause_time = 1000  # ms

    def get_image_sets(self):
        """
        Generate list of image set names.

        Note: Original code uses 'images1', 'images2', etc. (with 's')
        Different from memory test which uses 'image1', 'image2'
        """
        return [f"images{i}" for i in range(1, min(self.num_trials + 1, 33))]

    def load_image_pair(self, image_set):
        """
        Load a pair of images for binding test.

        Args:
            image_set: Name of image set (e.g., "images1")

        Returns:
            tuple: (image1_surface, image2_surface) or (None, None) if failed
        """
        try:
            img1_path = os.path.join(self.image_folder, f"{image_set}_1.jpg")
            img2_path = os.path.join(self.image_folder, f"{image_set}_2.jpg")

            if not os.path.exists(img1_path):
                print(f"⚠ Warning: Image not found: {img1_path}")
                return None, None
            if not os.path.exists(img2_path):
                print(f"⚠ Warning: Image not found: {img2_path}")
                return None, None

            img1 = pg.image.load(img1_path)
            img2 = pg.image.load(img2_path)

            return img1, img2
        except Exception as e:
            print(f"✗ Error loading images for {image_set}: {e}")
            return None, None

    def get_correct_answer(self, image_set):
        """
        Get correct answer for an image set.

        Args:
            image_set: Name of image set

        Returns:
            str: "Same" or "Different", or None if not found
        """
        if self.answer_df is None:
            return None

        try:
            answer = self.answer_df.loc[
                self.answer_df['Trial'] == image_set, 'Answer'
            ].values[0]
            return answer
        except (KeyError, IndexError) as e:
            print(f"⚠ Warning: No answer found for {image_set}")
            return None

    def show_instruction(self):
        """Display test instructions."""
        instruction = (
            "Binding Test: You will see two images.\n"
            "Focus on how the features (shapes, colors, positions) are bound together.\n"
            "Click 'Same' if they are identical.\n"
            "Click 'Different' if anything changed."
        )

        if self.parent and not self.auto_run:
            self.parent.display_screen.instruction = instruction
            self.parent.update_display()
            if self.parent.config.speech:
                self.parent.speak_text(
                    "You will see two images. "
                    "Tell me if the features are bound the same way or differently."
                )
        else:
            print("\n" + "="*50)
            print("Binding Test Instructions")
            print("="*50)
            print("Watch carefully how features are combined in each image.")
            print("Indicate if the second image has the same feature bindings.\n")

    def draw_buttons(self, surface):
        """Draw Same/Different buttons on surface."""
        if self.parent:
            width = int(self.parent.bottom_screen.get_width())
            height = int(self.parent.bottom_screen.get_height())
        else:
            width, height = 800, 600

        # Button dimensions
        button_width, button_height = 200, 60
        y_pos = height - 80

        # Same button (left)
        same_rect = pg.Rect(50, y_pos, button_width, button_height)
        pg.draw.rect(surface, (255, 255, 255), same_rect)
        pg.draw.rect(surface, (0, 0, 0), same_rect, 2)

        # Different button (right)
        diff_rect = pg.Rect(width - 250, y_pos, button_width, button_height)
        pg.draw.rect(surface, (255, 255, 255), diff_rect)
        pg.draw.rect(surface, (0, 0, 0), diff_rect, 2)

        # Text
        font = pg.font.Font(None, 36)
        same_text = font.render("Same", True, (0, 0, 0))
        diff_text = font.render("Different", True, (0, 0, 0))

        surface.blit(same_text, (same_rect.centerx - 40, same_rect.centery - 15))
        surface.blit(diff_text, (diff_rect.centerx - 50, diff_rect.centery - 15))

        return same_rect, diff_rect

    def show_image(self, image, duration_ms):
        """Display an image for a specified duration."""
        if self.parent:
            surface = self.parent.bottom_screen
        else:
            surface = pg.display.get_surface()

        # Clear and draw image centered
        grey = Colours.grey.value if INTEGRATED else (200, 200, 200)
        surface.fill(grey)

        img_rect = image.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2))
        surface.blit(image, img_rect)

        if self.parent:
            self.parent.update_display()
        else:
            pg.display.flip()

        pg.time.wait(duration_ms)

    def show_pause(self, duration_ms):
        """Show blank grey screen."""
        if self.parent:
            surface = self.parent.bottom_screen
        else:
            surface = pg.display.get_surface()

        grey = Colours.grey.value if INTEGRATED else (200, 200, 200)
        surface.fill(grey)

        if self.parent:
            self.parent.update_display()
        else:
            pg.display.flip()

        pg.time.wait(duration_ms)

    def show_image_and_get_response(self, image):
        """
        Show image and wait for Same/Different response.

        Returns:
            tuple: (response_string, response_time)
        """
        if self.parent:
            surface = self.parent.bottom_screen
        else:
            surface = pg.display.get_surface()

        # Draw image
        grey = Colours.grey.value if INTEGRATED else (200, 200, 200)
        surface.fill(grey)
        img_rect = image.get_rect(center=(surface.get_width() // 2, surface.get_height() // 2 - 50))
        surface.blit(image, img_rect)

        # Draw buttons
        same_rect, diff_rect = self.draw_buttons(surface)

        if self.parent:
            self.parent.update_display()
        else:
            pg.display.flip()

        # Wait for response
        response = None
        start_time = time.time()

        while response is None and self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                    return "Quit", 0

                elif event.type == pg.MOUSEBUTTONDOWN:
                    mouse_pos = pg.mouse.get_pos()

                    # Adjust for parent's bottom screen
                    if self.parent:
                        mouse_pos = self.parent.get_relative_mose_pos()

                    if same_rect.collidepoint(mouse_pos):
                        response = "Same"
                    elif diff_rect.collidepoint(mouse_pos):
                        response = "Different"

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        self.running = False
                        return "Quit", 0

        response_time = time.time() - start_time
        return response, response_time

    def run_trial(self, image_set, trial_num):
        """
        Run a single trial of the binding test.

        Args:
            image_set: Name of image set
            trial_num: Trial number (for display)

        Returns:
            dict: Trial results
        """
        # Load images
        img1, img2 = self.load_image_pair(image_set)
        if img1 is None or img2 is None:
            return None

        # Get correct answer
        correct_answer = self.get_correct_answer(image_set)

        # Show first image
        self.show_image(img1, self.image_display_time)

        # Pause (grey screen)
        self.show_pause(self.pause_time)

        # Show second image and get response
        response, response_time = self.show_image_and_get_response(img2)

        if response == "Quit":
            return None

        # Check if correct
        is_correct = (response == correct_answer) if correct_answer else None

        # Store results
        result = {
            "trial": trial_num,
            "image_set": image_set,
            "response": response,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "response_time": response_time
        }

        # Print feedback
        if is_correct is not None:
            status = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"Trial {trial_num}/{self.num_trials}: {status} - {response}")
        else:
            print(f"Trial {trial_num}/{self.num_trials}: {response} (no answer key)")

        return result

    def loop(self):
        """Main test execution loop."""
        self.running = True
        self.responses = []

        # Show instructions
        self.show_instruction()

        if not self.auto_run:
            pg.time.wait(3000)  # 3 second pause to read instructions

        # Run all trials
        image_sets = self.get_image_sets()

        for trial_num, image_set in enumerate(image_sets, 1):
            if not self.running:
                break

            result = self.run_trial(image_set, trial_num)
            if result:
                self.responses.append(result)

        # Calculate summary statistics
        correct_count = sum(1 for r in self.responses if r.get('is_correct') == True)
        total_with_answers = sum(1 for r in self.responses if r.get('is_correct') is not None)
        accuracy = (correct_count / total_with_answers * 100) if total_with_answers > 0 else 0
        avg_response_time = sum(r['response_time'] for r in self.responses) / len(self.responses) if self.responses else 0

        print(f"\n{'='*50}")
        print(f"Binding Test Complete!")
        print(f"{'='*50}")
        print(f"Accuracy: {correct_count}/{total_with_answers} ({accuracy:.1f}%)")
        print(f"Avg Response Time: {avg_response_time:.2f}s")
        print(f"{'='*50}\n")

        # Return results
        return {
            "responses": self.responses,
            "accuracy": round(accuracy, 2),
            "correct_count": correct_count,
            "total_trials": len(self.responses),
            "average_response_time": round(avg_response_time, 2)
        }

    def run_standalone(self):
        """Run in standalone mode for testing."""
        pg.init()
        screen = pg.display.set_mode((800, 600))
        pg.display.set_caption("Binding Test - Feature Binding Memory")

        results = self.loop()

        # Show results for a moment
        pg.time.wait(2000)
        pg.quit()

        return results


# Standalone testing
if __name__ == "__main__":
    print("="*50)
    print("Binding Test - Standalone Mode")
    print("="*50)
    print("Tests feature binding in visual memory\n")

    pg.init()
    test = BindingTest(num_trials=3)  # Just 3 trials for testing
    results = test.run_standalone()

    print("\nFinal Results:")
    print(f"Accuracy: {results['accuracy']}%")
    print(f"Correct: {results['correct_count']}/{results['total_trials']}")
    print(f"Avg Response Time: {results['average_response_time']:.2f}s")
