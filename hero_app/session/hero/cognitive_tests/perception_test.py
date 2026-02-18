"""
Perception Test Module - Visual perception and discrimination assessment

Shows images one at a time and asks if features are same or different.
Tests visual perception, discrimination, and decision-making speed.
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
        blue = pg.Color(0, 0, 255)
        red = pg.Color(255, 0, 0)

    class BlitLocation(Enum):
        centre = 8

    INTEGRATED = False


class PerceptionTest:
    """
    Visual perception test - assessing discrimination ability.

    Shows a series of images. Patient indicates whether features
    in the image are "Same" or "Different" based on what they perceive.
    Tracks accuracy and response time.
    """

    def __init__(self, parent=None, auto_run=False, num_trials=10,
                 image_folder="games/Images", answers_file="games/answers.xlsx"):
        """
        Initialize Perception Test.

        Args:
            parent: Parent Consultation object
            auto_run: If True, runs automatically
            num_trials: Number of images to show (default: 10)
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
        self.response_times = []
        self.score = 0
        self.current_trial = 0

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

    def get_image_files(self):
        """
        Get list of image file paths.

        Returns:
            list: Paths to image files (1.jpg, 2.jpg, ..., 10.jpg)
        """
        return [
            os.path.join(self.image_folder, f"{i}.jpg")
            for i in range(1, min(self.num_trials + 1, 11))
        ]

    def load_image(self, image_path):
        """
        Load and scale an image.

        Args:
            image_path: Path to image file

        Returns:
            pygame.Surface or None if failed
        """
        try:
            if not os.path.exists(image_path):
                print(f"⚠ Warning: Image not found: {image_path}")
                return None

            image = pg.image.load(image_path)

            # Scale to fit screen
            if self.parent:
                width = int(self.parent.bottom_screen.get_width())
                height = int(self.parent.bottom_screen.get_height())
            else:
                width, height = 800, 600

            # Scale image to fit, maintaining aspect ratio
            image = pg.transform.scale(image, (width, height - 100))  # Leave room for buttons

            return image
        except Exception as e:
            print(f"✗ Error loading image {image_path}: {e}")
            return None

    def get_correct_answer(self, trial_index):
        """
        Get correct answer for a trial.

        Args:
            trial_index: Trial index (0-based)

        Returns:
            str: "Same" or "Different", or None if not found
        """
        if self.answer_df is None:
            return None

        try:
            answer = self.answer_df.loc[trial_index, 'Answer']
            return answer
        except (KeyError, IndexError):
            print(f"⚠ Warning: No answer found for trial {trial_index}")
            return None

    def show_instruction(self):
        """Display test instructions."""
        instruction = (
            "Perception Test\n"
            "Look at each image carefully.\n"
            "Click 'Same' if elements appear identical.\n"
            "Click 'Different' if elements are not identical."
        )

        if self.parent and not self.auto_run:
            self.parent.display_screen.instruction = instruction
            self.parent.update_display()
            if self.parent.config.speech:
                self.parent.speak_text(
                    "Look at each image carefully. "
                    "Tell me if the elements are the same or different."
                )
        else:
            print("\n" + "="*50)
            print("Perception Test Instructions")
            print("="*50)
            print("Look at each image.")
            print("Click 'Same' if elements are identical.")
            print("Click 'Different' if they are not.\n")

    def show_ready_screen(self):
        """Show 'Are you ready?' screen."""
        if not self.auto_run and not self.parent:
            # Only show in standalone mode
            screen = pg.display.get_surface()
            font = pg.font.Font(None, 36)

            waiting = True
            next_button = pg.Rect(700, 500, 80, 50)

            while waiting:
                screen.fill((255, 255, 255))

                ready_text = font.render("Are you ready?", True, (0, 0, 0))
                ready_text2 = font.render("Click 'Next' to start", True, (0, 0, 0))
                screen.blit(ready_text, (250, 200))
                screen.blit(ready_text2, (220, 250))

                pg.draw.rect(screen, (0, 0, 255), next_button)
                next_text = font.render("Next", True, (255, 255, 255))
                screen.blit(next_text, (720, 515))

                pg.display.flip()

                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        self.running = False
                        return
                    elif event.type == pg.MOUSEBUTTONDOWN:
                        if next_button.collidepoint(pg.mouse.get_pos()):
                            waiting = False

    def draw_buttons(self, surface):
        """
        Draw Same/Different buttons on surface.

        Returns:
            tuple: (same_rect, different_rect)
        """
        if self.parent:
            width = int(self.parent.bottom_screen.get_width())
            height = int(self.parent.bottom_screen.get_height())
        else:
            width, height = 800, 600

        # Button positions
        same_rect = pg.Rect(100, height - 80, 100, 50)
        diff_rect = pg.Rect(width - 240, height - 80, 140, 50)

        # Draw buttons
        blue = Colours.blue.value if INTEGRATED else (0, 0, 255)
        red = Colours.red.value if INTEGRATED else (255, 0, 0)
        white = Colours.white.value if INTEGRATED else (255, 255, 255)

        pg.draw.rect(surface, blue, same_rect)
        pg.draw.rect(surface, red, diff_rect)

        # Text
        font = pg.font.Font(None, 36)
        same_text = font.render("Same", True, white)
        diff_text = font.render("Different", True, white)

        surface.blit(same_text, (same_rect.x + 20, same_rect.y + 15))
        surface.blit(diff_text, (diff_rect.x + 20, diff_rect.y + 15))

        return same_rect, diff_rect

    def draw_score(self, surface):
        """Draw current score on surface."""
        font = pg.font.Font(None, 36)
        black = Colours.black.value if INTEGRATED else (0, 0, 0)

        score_text = font.render(f"Score: {self.score}/{self.current_trial}", True, black)
        surface.blit(score_text, (10, 10))

    def run_trial(self, image_path, trial_num):
        """
        Run a single trial.

        Args:
            image_path: Path to image file
            trial_num: Trial number (1-based for display)

        Returns:
            dict: Trial results
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None

        # Get correct answer
        correct_answer = self.get_correct_answer(trial_num - 1)  # 0-indexed

        # Show image and get response
        response, response_time = self.show_image_and_get_response(image)

        if response == "Quit":
            return None

        # Check if correct
        is_correct = (response == correct_answer) if correct_answer else None

        if is_correct:
            self.score += 1

        # Store results
        result = {
            "trial": trial_num,
            "image": os.path.basename(image_path),
            "response": response,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "response_time": response_time,
            "score_after_trial": self.score
        }

        # Print feedback
        if is_correct is not None:
            status = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"Trial {trial_num}/{self.num_trials}: {status} - {response} ({response_time:.2f}s)")
        else:
            print(f"Trial {trial_num}/{self.num_trials}: {response} ({response_time:.2f}s) [no answer key]")

        return result

    def show_image_and_get_response(self, image):
        """
        Show image and wait for response.

        Args:
            image: pygame.Surface to display

        Returns:
            tuple: (response_string, response_time)
        """
        if self.parent:
            surface = self.parent.bottom_screen
        else:
            surface = pg.display.get_surface()

        # Start timer
        start_time = time.time()

        response = None

        while response is None and self.running:
            # Clear screen
            white = Colours.white.value if INTEGRATED else (255, 255, 255)
            surface.fill(white)

            # Draw image at top
            surface.blit(image, (0, 0))

            # Draw buttons
            same_rect, diff_rect = self.draw_buttons(surface)

            # Draw score
            self.draw_score(surface)

            # Update display
            if self.parent:
                self.parent.update_display()
            else:
                pg.display.flip()

            # Handle events
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

    def loop(self):
        """Main test execution loop."""
        self.running = True
        self.responses = []
        self.score = 0
        self.current_trial = 0

        # Show instructions
        self.show_instruction()

        if not self.auto_run:
            pg.time.wait(2000)

        # Show ready screen
        self.show_ready_screen()

        # Get image files
        image_files = self.get_image_files()

        # Run all trials
        for trial_num, image_path in enumerate(image_files, 1):
            if not self.running:
                break

            self.current_trial = trial_num
            result = self.run_trial(image_path, trial_num)

            if result:
                self.responses.append(result)

        # Calculate summary statistics
        correct_count = sum(1 for r in self.responses if r.get('is_correct') == True)
        total_with_answers = sum(1 for r in self.responses if r.get('is_correct') is not None)
        accuracy = (correct_count / total_with_answers * 100) if total_with_answers > 0 else 0
        avg_response_time = sum(r['response_time'] for r in self.responses) / len(self.responses) if self.responses else 0

        print(f"\n{'='*50}")
        print(f"Perception Test Complete!")
        print(f"{'='*50}")
        print(f"Final Score: {self.score}/{len(self.responses)}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Avg Response Time: {avg_response_time:.2f}s")
        print(f"{'='*50}\n")

        # Return results
        return {
            "responses": self.responses,
            "score": self.score,
            "accuracy": round(accuracy, 2),
            "correct_count": correct_count,
            "total_trials": len(self.responses),
            "average_response_time": round(avg_response_time, 2)
        }

    def run_standalone(self):
        """Run in standalone mode for testing."""
        pg.init()
        screen = pg.display.set_mode((800, 600))
        pg.display.set_caption("Perception Test")
        clock = pg.time.Clock()

        results = self.loop()

        # Show final results
        if results and self.running:
            screen.fill((255, 255, 255))
            font = pg.font.Font(None, 48)

            final_text = font.render(f"Final Score: {results['score']}/{results['total_trials']}", True, (0, 0, 0))
            accuracy_text = font.render(f"Accuracy: {results['accuracy']}%", True, (0, 0, 0))

            screen.blit(final_text, (200, 250))
            screen.blit(accuracy_text, (220, 320))

            pg.display.flip()
            pg.time.wait(3000)

        pg.quit()
        return results


# Standalone testing
if __name__ == "__main__":
    print("="*50)
    print("Perception Test - Standalone Mode")
    print("="*50)
    print("Visual perception and discrimination test\n")

    pg.init()
    test = PerceptionTest(num_trials=3)  # Just 3 trials for testing
    results = test.run_standalone()

    if results:
        print("\nFinal Results:")
        print(f"Score: {results['score']}/{results['total_trials']}")
        print(f"Accuracy: {results['accuracy']}%")
        print(f"Avg Response Time: {results['average_response_time']:.2f}s")
