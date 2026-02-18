"""
Consultation Orchestrator - Main consultation flow controller

Manages the complete patient assessment workflow including:
- User authentication
- Cognitive test battery
- Affective computing
- Data collection and storage
"""

import datetime
import json
import os
import re
import shutil
import string
import time
from datetime import date

import cv2
import gtts
import numpy as np
import pandas as pd
import pygame as pg

# Hero imports
from hero.data.db_access import DBClient
from hero.consultation.config import ConsultConfig, get_mongo_client
from hero.consultation.avatar import Avatar
from hero.consultation.display_screen import DisplayScreen
from hero.consultation.touch_screen import TouchScreen, GameObjects, GameButton
from hero.consultation.utils import take_screenshot, NpEncoder, Buttons, ButtonModule
from hero.consultation.screen import Colours, Fonts

# Cognitive test imports
from hero.cognitive_tests.clock_draw import ClockDraw
from hero.cognitive_tests.perceived_stress_score import PSS
from hero.cognitive_tests.shape_searcher import ShapeSearcher
from hero.cognitive_tests.spiral_test import SpiralTest
from hero.cognitive_tests.visual_attention_test import VisualAttentionTest
from hero.cognitive_tests.wisconsin_card_test import CardGame
from hero.consultation.login_screen import LoginScreen

# Affective computing
try:
    from hero.affective_computing.affective_computing_pi import AffectiveModulePi
    AFFECTIVE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Warning: Affective computing not available: {e}")
    AFFECTIVE_AVAILABLE = False
    AffectiveModulePi = None

class User:
    """Patient user model."""

    def __init__(self, name, age, user_id):
        self.id = user_id
        self.name = name
        self.age = age


class Consultation:
    """
    Main consultation orchestrator.

    Manages the complete assessment workflow including authentication,
    cognitive tests, affective computing, and data storage.
    """

    def __init__(self, enable_speech=True, scale=1, pi=True, authenticate=True,
                 seamless=True, username=None, password=None, consult_date=None,
                 auto_run=False, wct_turns=20, pss_questions=10, db_client=None,
                 local=True):
        """
        Initialize Consultation.

        Args:
            enable_speech: Enable text-to-speech
            scale: Display scale factor
            pi: Running on Raspberry Pi
            authenticate: Require user login
            seamless: Auto-advance through tests
            username: Pre-filled username (for testing)
            password: Pre-filled password (for testing)
            consult_date: Override consultation date
            auto_run: Automatically run tests (for testing)
            wct_turns: Number of Wisconsin Card Test turns
            pss_questions: Number of PSS questions
            db_client: Database client (if not local)
            local: Save locally vs remote database
        """
        # User authentication
        self.authenticate_user = authenticate
        self.user = None

        # Configuration
        self.config = ConsultConfig(speech=enable_speech)
        self.auto_run = auto_run
        self.seamless = seamless
        self.pi = pi
        self.local = local

        # Audio temp directory
        self.audio_temp_dir = "temp/question_audio"
        os.makedirs(self.audio_temp_dir, exist_ok=True)

        # UPDATED: Resources directory
        self.resources_dir = "hero/consultation/resources"

        # Display setup
        self.display_size = pg.Vector2(1024, 600) * scale

        if not pg.get_init():
            pg.init()
        if not pg.font.get_init():
            pg.font.init()

            # NOW create fonts
        self.fonts = Fonts()

        # Load user data
        if os.path.exists("data/user_data.csv"):
            self.all_user_data = pd.read_csv("data/user_data.csv")
            self.all_user_data = self.all_user_data.set_index("Username")
        else:
            self.all_user_data = None

        # Initialize pygame window
        if pi:
            self.window = pg.display.set_mode(
                (self.display_size.x, self.display_size.y * 2),
                pg.NOFRAME | pg.SRCALPHA
            )
        else:
            self.window = pg.display.set_mode(
                (self.display_size.x, self.display_size.y * 2),
                pg.SRCALPHA
            )

        # Create dual screens
        self.top_screen = self.window.subsurface(((0, 0), self.display_size))
        self.bottom_screen = self.window.subsurface(
            (0, self.display_size.y),
            self.display_size
        )

        # UI components
        self.fonts = Fonts()
        self.display_screen = DisplayScreen(self.top_screen.get_size())
        self.touch_screen = TouchScreen(self.bottom_screen.get_size())

        # Buttons
        button_size = pg.Vector2(300, 200)
        self.quit_button = GameButton(
            (10, 10),
            pg.Vector2(70, 50),
            id=2,
            text="QUIT",
            colour=Colours.red
        )
        self.main_button = GameButton(
            (self.display_size - button_size) / 2,
            button_size,
            id=1,
            text="Start"
        )

        # Avatar
        self.avatar = Avatar(size=(self.display_size.y * 0.7, self.display_size.y * 0.7))
        self.display_screen.avatar = self.avatar

        # Hardware buttons
        self.button_module = ButtonModule(pi)

        # Initialize test modules
        self.pss_question_count = pss_questions
        self.modules = {
            "Shapes": ShapeSearcher(parent=self, auto_run=auto_run),
            "Spiral": SpiralTest(
                turns=3,
                spiral_size=self.display_size.y * 0.9,
                parent=self,
                auto_run=auto_run
            ),
            "VAT": VisualAttentionTest(
                parent=self,
                grid_size=(self.display_size.y * 0.9, self.display_size.y * 0.9),
                auto_run=auto_run
            ),
            "WCT": CardGame(
                parent=self,
                max_turns=wct_turns,
                auto_run=auto_run
            ),
            "PSS": PSS(
                parent=self,
                question_count=self.pss_question_count,
                auto_run=auto_run,
                preload_audio=False
            ),
            "Clock": ClockDraw(parent=self, auto_run=auto_run),
            "Login": LoginScreen(
                parent=self,
                username=username,
                password=password,
                auto_run=auto_run
            ),
        }

        # Add affective computing only if available
        if AFFECTIVE_AVAILABLE:
            self.modules["Affective"] = AffectiveModulePi(
                parent=self,
                pi=pi,
                cleanse_files=False,
                auto_run=auto_run
            )

        # Test execution order
        self.module_order = ["Login", "Spiral", "Clock", "Shapes", "VAT", "WCT", "PSS"]
        self.module_idx = 0

        # State
        self.running = True
        self.output = None

        # Consultation metadata
        self.id = self.generate_unique_id()
        self.date = consult_date if consult_date else date.today()

        # Database
        if not local:
            if db_client:
                self.db_client = db_client
            else:
                # Try to get MongoDB client
                mongo_client = get_mongo_client()
                if mongo_client:
                    self.db_client = DBClient()
                else:
                    print("⚠ Warning: Could not connect to database, forcing local mode")
                    self.local = True
                    self.db_client = None
        else:
            self.db_client = None

        # Hide mouse on Pi
        if pi:
            pg.mouse.set_visible(False)

    def generate_unique_id(self):
        """Generate unique consultation ID."""
        letters = pd.Series(list(string.ascii_lowercase))[
                      np.random.permutation(26)
                  ][:10].values
        numbers = np.random.permutation(10)[:5]
        num_pos = np.sort(np.random.permutation(range(1, 15))[:5])

        for idx, num in zip(num_pos, numbers):
            letters = np.insert(letters, idx, num)

        return "".join(str(elem) for elem in letters)

    def update_display(self, display_screen=None, touch_screen=None):
        """Update display with current screen states."""
        if display_screen is None:
            display_screen = self.display_screen
        if touch_screen is None:
            touch_screen = self.touch_screen

        self.top_screen.blit(display_screen.get_surface(), (0, 0))
        self.bottom_screen.blit(touch_screen.get_surface(), (0, 0))
        pg.display.flip()

    def speak_text(self, text, visual=True, display_screen=None, touch_screen=None):
        """
        Speak text using text-to-speech with avatar animation.

        Args:
            text: Text to speak
            visual: Show avatar mouth animation
            display_screen: Display screen to update
            touch_screen: Touch screen to update
        """
        if self.auto_run:
            return
        if not self.config.speech:
            return

        if not display_screen:
            display_screen = self.display_screen
        if not touch_screen:
            touch_screen = self.touch_screen

        # Convert text to phonetic mouth positions
        mouth_ids = self._text_to_mouth_ids(text)

        # Generate and play audio
        question_audio = gtts.gTTS(text=text, lang='en', tld='com.au', slow=False)
        question_audio_file = 'temp/question_audio/tempsave.mp3'
        question_audio.save(question_audio_file)

        pg.mixer.music.load(question_audio_file)
        pg.mixer.music.play()

        # Animate avatar mouth
        if visual:
            temp_instruction = display_screen.instruction
            display_screen.instruction = None

            mouth_idx = 0
            start = time.monotonic()

            while pg.mixer.music.get_busy():
                if time.monotonic() - start > 0.15:
                    try:
                        display_screen.avatar.mouth_idx = mouth_ids[mouth_idx]
                        self.update_display(display_screen, touch_screen)
                        start = time.monotonic()
                        mouth_idx += 1
                    except IndexError:
                        pass

            display_screen.avatar.mouth_idx = 0
            display_screen.instruction = temp_instruction
            self.update_display(display_screen, touch_screen)

    def _text_to_mouth_ids(self, text):
        """Convert text to phonetic mouth animation IDs."""
        text_index = text.lower()
        text_index = text_index.replace("?", "").replace("!", "")
        text_index = text_index.replace(""", "").replace(""", "")
        text_index = text_index.replace(" ", "0 ").replace(".", "0 ").replace(",", "0 ")

        # Phonetic replacements
        rep_1 = {"ee": "7 ", "th": "8 ", "sh": "9 ", "ch": "9 "}
        rep_2 = {
            "a": "0 ", "e": "0 ", "i": "0 ", "o": "1 ",
            "c": "2 ", "d": "2 ", "n": "2 ", "s": "2 ",
            "t": "2 ", "x": "2 ", "y": "2 ", "z": "2 ",
            "g": "3 ", "k": "3 ", "l": "4 ",
            "b": "5 ", "m": "5 ", "p": "5 ",
            "f": "6 ", "v": "6 ", "j": "9 ",
            "u": "10 ", "q": "11 ", "w": "11 ", "h": ""
        }

        # Apply replacements
        for rep in [rep_1, rep_2]:
            escaped = dict((re.escape(k), v) for k, v in rep.items())
            regex = re.compile("|".join(escaped.keys()))
            text_index = regex.sub(
                lambda m, e=escaped: e[re.escape(m.group(0))],
                text_index
            )

        # Extract mouth IDs
        mouth_ids = []
        for num in text_index.strip().split(" "):
            try:
                mouth_ids.append(int(num))
            except ValueError:
                pass

        return mouth_ids

    def get_relative_mose_pos(self):
        """Get mouse position relative to bottom screen."""
        return pg.Vector2(pg.mouse.get_pos()) - pg.Vector2(0, self.display_size.y)

    def take_screenshot(self, filename=None):
        """Take screenshot of current display."""
        print("Taking Screenshot")
        img_array = pg.surfarray.array3d(self.window)
        img_array = cv2.transpose(img_array)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if filename is None:
            filename = datetime.datetime.now()

        os.makedirs("screenshots", exist_ok=True)
        cv2.imwrite(f"screenshots/{filename}.png", img_array)

    def entry_sequence(self):
        """Initial consultation setup and authentication."""
        # User login
        if self.authenticate_user:
            self.user = self.modules["Login"].loop()
            self.speak_text(f"Welcome back {self.user.name}")

        # Setup UI for non-seamless mode
        if not self.seamless:
            self.touch_screen.sprites = GameObjects([self.quit_button, self.main_button])
            self.display_screen.instruction = "Click the button to start"
            self.update_display()

    def exit_sequence(self):
        """Finalize consultation and save data."""
        self.speak_text("The consultation is now complete. Thank you for your time")

        # Show processing screen
        self.display_screen.power_off = True
        self.touch_screen.power_off = True

        proc_surf = self.fonts.normal.render(
            "Processing...",
            True,
            Colours.hero_blue.value
        )

        disp_copy = self.display_screen.power_off_surface.copy()
        self.display_screen.power_off_surface.blit(
            proc_surf,
            (self.display_size - proc_surf.get_size()) / 2 + pg.Vector2(0, 100)
        )
        self.update_display()

        # Process affective computing data
        if "Affective" in self.modules:
            self.modules["Affective"].exit_sequence()

        self.display_screen.power_off_surface = disp_copy
        self.update_display()

        # Compile results
        results = self._compile_results()

        # Save results
        self._save_results(results)

        # Cleanup
        shutil.rmtree("temp/question_audio")

        print(f"Successfully completed consultation {self.id}")

    def _compile_results(self):
        """Compile all test results into output format."""
        # PSS processing
        pss_answers = np.array(self.modules["PSS"].answers)
        if pss_answers.size > 0:
            pss_reverse_idx = np.array([3, 4, 6, 7])
            pss_reverse_idx = pss_reverse_idx[pss_reverse_idx < self.pss_question_count]
            pss_answers[pss_reverse_idx] = 4 - pss_answers[pss_reverse_idx]

        pss_data = {"answers": pss_answers.tolist()}

        # Wisconsin Card Test
        wct_data = {
            "answers": self.modules["WCT"].engine.answers,
            "change_ids": self.modules["WCT"].engine.new_rule_ids
        }

        # Visual Attention Test
        vat_data = {
            "answers": self.modules["VAT"].answers,
            "times": self.modules["VAT"].answer_times
        }

        # Clock Draw
        clock_data = {"angle_errors": self.modules["Clock"].angle_errors}

        # Shape Searcher
        shape_data = {
            "scores": self.modules["Shapes"].scores,
            "question_counts": self.modules["Shapes"].question_counts,
            "answer_times": self.modules["Shapes"].answer_times
        }

        # Spiral Test
        spiral_data = {
            "classification": int(self.modules["Spiral"].classification),
            "value": self.modules["Spiral"].prediction
        }

        # Affective Computing
        affective_data = {}  # self.modules["Affective"].label_data

        # Compile output
        user_id = self.user.id if self.user else None

        return {
            "consult_id": self.id,
            "user_id": int(user_id) if user_id else None,
            "consult_time": self.date.strftime("%Y-%m-%d"),
            "consult_data": {
                "pss": pss_data,
                "wct": wct_data,
                "vat": vat_data,
                "clock": clock_data,
                "shape": shape_data,
                "spiral": spiral_data,
                "affective": affective_data
            }
        }

    def _save_results(self, results):
        """Save consultation results to local or remote storage."""
        self.output = results

        if not self.user:
            return

        if self.local:
            # Save locally
            base_path = "data"
            record_path = os.path.join(base_path, "consult_records")
            os.makedirs(record_path, exist_ok=True)

            user_path = os.path.join(record_path, f"user_{self.user.id}")
            os.makedirs(user_path, exist_ok=True)

            consult_path = os.path.join(user_path, f"consult_{self.id}")

            with open(consult_path, "w") as f:
                json.dump(self.output, f, cls=NpEncoder, indent=4)
        else:
            # Upload to database
            self.db_client.upload_consult(self.output)

    def loop(self, infinite=False):
        """
        Main consultation loop.

        Args:
            infinite: Loop through tests indefinitely (for demos)
        """
        self.entry_sequence()

        while self.running:
            if self.seamless:
                # Auto-run all tests
                for module in self.module_order:
                    self.modules[module].loop()
                    self.update_display()
                self.running = False

            else:
                # Manual progression
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_ESCAPE:
                            self.running = False
                        elif event.key == pg.K_s:
                            self.take_screenshot()

                    elif event.type == pg.MOUSEBUTTONDOWN:
                        button_id = self.touch_screen.click_test(
                            self.get_relative_mose_pos()
                        )

                        if button_id == 1:  # Start button
                            self.touch_screen.kill_sprites()
                            self.update_display()

                            # Run current module
                            module = self.modules[self.module_order[self.module_idx]]
                            module.running = True
                            module.loop()

                            # Progress to next module
                            self.display_screen.instruction = "Click the button to start"
                            self.update_display()

                            if infinite:
                                self.module_idx = (self.module_idx + 1) % len(self.modules)
                            else:
                                self.module_idx += 1
                                if self.module_idx >= len(self.module_order):
                                    self.running = False

                            self.touch_screen.sprites = GameObjects([
                                self.quit_button,
                                self.main_button
                            ])
                            self.update_display()

                        elif button_id == 2:  # Quit button
                            self.running = False

                    elif event.type == pg.QUIT:
                        self.running = False

        self.exit_sequence()


# Standalone testing
if __name__ == "__main__":
    os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
    pg.init()
    pg.font.init()
    pg.event.pump()

    consult = Consultation(
        pi=False,
        authenticate=True,
        seamless=True,
        auto_run=True,
        username="user k",
        password="pass",
        pss_questions=2,
        local=True  # Save locally for testing
    )

    consult.loop()
