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
import logging
import os
import re
import shutil
import string
import time
from datetime import date, timezone

logger = logging.getLogger(__name__)

import gtts
import numpy as np
import pandas as pd
import pygame as pg

# Hero imports
from hero.consultation.config import ConsultConfig
from hero.consultation.avatar import Avatar
from hero.consultation.display_screen import DisplayScreen
from hero.consultation.touch_screen import TouchScreen, GameObjects, GameButton
from hero.consultation.utils import take_screenshot, NpEncoder, Buttons, ButtonModule
from hero.consultation.screen import Colours, Fonts

# Cognitive test imports
from hero.cognitive_tests.shape_searcher import ShapeSearcher
from hero.cognitive_tests.spiral_test import SpiralTest
from hero.cognitive_tests.memory_game import MemoryGame
from hero.cognitive_tests.trail_making_test import TrailMakingTest
from hero.consultation.login_screen import LoginScreen

# Affective computing
try:
    from hero.affective_computing.affective_computing_pi import AffectiveModulePi
    AFFECTIVE_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Warning: Affective computing not available: {e}")
    AFFECTIVE_AVAILABLE = False
    AffectiveModulePi = None

# Screenshot (cv2 optional)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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
                 local=True, db=None, clock=None, on_session_start=None):
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
            wct_turns: Legacy, unused
            pss_questions: Legacy, unused
            db_client: Legacy, unused
            local: Save locally as JSON backup
            db: HeroDB instance for PostgreSQL integration
            clock: CentralClock instance for synchronized timestamps
                   (shared with SensorCoordinator so all data is on the same clock)
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

        # HERO DB integration
        self.db = db
        self.session_id = None

        # Central clock — falls back to datetime.now() if not provided
        self.clock = clock
        self.on_session_start = on_session_start

        # Audio temp directory
        self.audio_temp_dir = "temp/question_audio"
        os.makedirs(self.audio_temp_dir, exist_ok=True)

        # Resources directory
        self.resources_dir = "hero/consultation/resources"

        # Display setup
        self.display_size = pg.Vector2(1024, 600) * scale

        if not pg.get_init():
            pg.init()
        if not pg.font.get_init():
            pg.font.init()

        self.fonts = Fonts()

        # Load user data (CSV fallback for login)
        if os.path.exists("data/user_data.csv"):
            self.all_user_data = pd.read_csv("data/user_data.csv")
            self.all_user_data = self.all_user_data.set_index("Username")
        else:
            self.all_user_data = None

        # Initialize pygame window — spans both physical screens (1024x1200)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "1029,600"
        if pi:
            self.window = pg.display.set_mode(
                (int(self.display_size.x), int(self.display_size.y * 2)),
                pg.NOFRAME | pg.SRCALPHA
            )
        else:
            self.window = pg.display.set_mode(
                (int(self.display_size.x), int(self.display_size.y * 2)),
                pg.SRCALPHA
            )

        # Force window to correct position on combined display
        try:
            import ctypes
            wm_info = pg.display.get_wm_info()
            print(f"WM info keys: {list(wm_info.keys())}")
            if 'window' in wm_info:
                import subprocess
                subprocess.Popen(['bash', '-c',
                    'sleep 0.5 && wmctrl -r :ACTIVE: -e 0,1029,600,-1,-1'
                ])
        except Exception as e:
            print(f"Window positioning failed: {e}")

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
        self.modules = {
            "Login": LoginScreen(
                parent=self,
                username=username,
                password=password,
                auto_run=auto_run
            ),
            "Spiral": SpiralTest(
                turns=3,
                spiral_size=self.display_size.y * 0.9,
                parent=self,
                auto_run=auto_run
            ),
            "Shapes": ShapeSearcher(parent=self, auto_run=auto_run),
            "Memory": MemoryGame(parent=self, auto_run=auto_run),
            "Trail": TrailMakingTest(parent=self, auto_run=auto_run),
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
        self.module_order = ["Login", "Spiral", "Trail", "Shapes", "Memory"]
        self.module_idx = 0

        # State
        self.running = True
        self.output = None

        # Consultation metadata
        self.id = self.generate_unique_id()
        self.date = consult_date if consult_date else date.today()

        # Hide mouse on Pi
        if pi:
            pg.mouse.set_visible(False)

    # ------------------------------------------------------------------
    # Timestamp helper — uses central clock if available
    # ------------------------------------------------------------------

    def now(self):
        """
        Get current timestamp.
        Uses CentralClock if provided (shared with sensors),
        otherwise falls back to system clock.
        """
        if self.clock:
            return self.clock.now()
        return datetime.datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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

        mouth_ids = self._text_to_mouth_ids(text)

        question_audio = gtts.gTTS(text=text, lang='en', tld='com.au', slow=False)
        question_audio_file = 'temp/question_audio/tempsave.mp3'
        question_audio.save(question_audio_file)

        pg.mixer.music.load(question_audio_file)
        pg.mixer.music.play()

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
        text_index = text_index.replace("\u201c", "").replace("\u201d", "")
        text_index = text_index.replace(" ", "0 ").replace(".", "0 ").replace(",", "0 ")

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

        for rep in [rep_1, rep_2]:
            escaped = dict((re.escape(k), v) for k, v in rep.items())
            regex = re.compile("|".join(escaped.keys()))
            text_index = regex.sub(
                lambda m, e=escaped: e[re.escape(m.group(0))],
                text_index
            )

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
        if not CV2_AVAILABLE:

            return


        img_array = pg.surfarray.array3d(self.window)
        img_array = cv2.transpose(img_array)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if filename is None:
            filename = datetime.datetime.now()

        os.makedirs("screenshots", exist_ok=True)
        cv2.imwrite(f"screenshots/{filename}.png", img_array)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _start_db_session(self):
        """Create a session in the DB once the user is known."""
        if not self.db or not self.user:
            return
        try:
            self.session_id = self.db.start_session(
                user_id=self.user.id,
                notes=f"Consultation {self.id}"
            )
            logger.info("✓ DB session started: {self.session_id}")
            if self.on_session_start:
                self.on_session_start(self.session_id)
        except Exception as e:
            print(f"⚠ Could not start DB session: {e}")

    def _save_game_to_db(self, module_name, game_number, started_at, completed_at):
        """Save a single game result to the DB."""
        if not self.db or not self.session_id:
            return
        if module_name == "Login":
            return

        module = self.modules[module_name]
        results = getattr(module, 'results', {}) or {}

        # Per-game field mappings — each game now has a consistent results dict
        if module_name == "Spiral":
            final_score     = None
            max_score       = None
            accuracy        = None
            correct         = None
            incorrect       = None
            avg_rt_ms       = None

        elif module_name == "Trail":
            final_score     = None
            max_score       = None
            accuracy        = None
            correct         = None
            incorrect       = results.get('errors')
            avg_rt_ms       = None

        elif module_name == "Shapes":
            total_q         = results.get('total_questions', 0)
            correct         = results.get('correct', 0)
            final_score     = correct
            max_score       = total_q
            accuracy        = results.get('accuracy_percent')
            incorrect       = results.get('incorrect', 0)
            rt_s            = results.get('avg_reaction_time_s')
            avg_rt_ms       = round(rt_s * 1000, 1) if rt_s else None

        elif module_name == "Memory":
            final_score     = results.get('score')
            max_score       = results.get('total_trials')
            accuracy        = results.get('accuracy_percent')
            correct         = results.get('score')
            incorrect       = results.get('incorrect_count')
            avg_rt_ms       = None

        else:
            final_score     = results.get('score')
            max_score       = results.get('total_trials') or results.get('max_score')
            accuracy        = results.get('accuracy') or results.get('accuracy_percent')
            correct         = results.get('score') or results.get('correct_answers')
            incorrect       = results.get('incorrect_answers')
            avg_rt_ms       = results.get('average_reaction_time_ms')

        try:
            self.db.save_game_result(
                session_id=self.session_id,
                game_name=module_name,
                game_number=game_number,
                started_at=started_at,
                completed_at=completed_at,
                final_score=final_score,
                max_score=max_score,
                accuracy_percent=accuracy,
                correct_answers=correct,
                incorrect_answers=incorrect,
                average_reaction_time_ms=avg_rt_ms,
                game_data=results,
                completion_status='completed'
            )

        except Exception as e:
            logger.warning(f"Error: {e}")

    
    def _log_game_event(self, module_name, game_number, event_type, timestamp):
        """Log a game_start or game_end event."""
        if not self.db or not self.session_id:
            return
        if module_name == "Login":
            return
        try:
            self.db.log_event(
                session_id=self.session_id,
                event_type=event_type,
                event_category='game',
                game_name=module_name,
                game_number=game_number,
                event_data={'consultation_id': self.id},
            )
        except Exception as e:
            logger.warning(f"Error: {e}")

    
    def _run_eye_tracking_calibration(self):
        """
        Run 9-point eye tracking calibration after login.
        Saves polynomial model coefficients to DB for use by the sensor pipeline.
        Shows a simple status screen on the pygame display before/after.
        Skips gracefully if camera not available.
        """
        if not self.session_id:

            return

        # Show "calibrating..." screen
        self.display_screen.refresh()
        self.display_screen.instruction = "Eye Tracking Calibration — starting..."
        self.update_display()

        try:
            from hero_system.sensors.eye_tracking.calibrator import EyeTrackingCalibrator
            from hero_system.sensors.eye_tracking.config import EyeTrackingConfig
            from hero_core.database.models.connection import get_db_connection

            _, db_session = get_db_connection(
                host='localhost', port=5432, user='postgres',
                password='pgdbadmin', dbname='hero_db'
            )

            config = EyeTrackingConfig.for_calibration()
            calibrator = EyeTrackingCalibrator(
                session_id=self.session_id,
                db_session=db_session,
                config=config,
            )

            # Pygame must yield display to OpenCV for calibration window
            pg.display.iconify()

            calibrator.start()
            success = calibrator.run_calibration()

            if success:
                calibrator.save_to_database()

            calibrator.stop()
            db_session.close()

        except Exception as e:
            print(f"⚠ Eye tracking calibration failed (continuing without): {e}")

        finally:
            # Restore pygame window
            pg.display.set_mode(
                (self.display_size.x, self.display_size.y * 2),
                pg.NOFRAME | pg.SRCALPHA if self.pi else pg.SRCALPHA
            )
            self.top_screen    = self.window.subsurface(((0, 0), self.display_size))
            self.bottom_screen = self.window.subsurface(
                (0, self.display_size.y), self.display_size
            )
            self.display_screen.instruction = "Calibration complete — starting tests"
            self.update_display()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def entry_sequence(self):
        """Initial consultation setup."""
        if not self.seamless:
            self.touch_screen.sprites = GameObjects([self.quit_button, self.main_button])
            self.display_screen.instruction = "Click the button to start"
            self.update_display()

    def exit_sequence(self):
        """Finalize consultation and save data."""
        self.speak_text("The consultation is now complete. Thank you for your time")

        self.display_screen.power_off = True
        self.touch_screen.power_off = True
        self.update_display()

        # Wait for a screen press or any key before closing
        pg.event.clear()
        waiting = True
        while waiting:
            for event in pg.event.get():
                if event.type in (pg.MOUSEBUTTONDOWN, pg.FINGERDOWN, pg.KEYDOWN):
                    waiting = False
            time.sleep(0.05)

        if "Affective" in self.modules:
            self.modules["Affective"].exit_sequence()

        # Save local JSON backup
        if self.local:
            results = self._compile_results()
            self._save_results_local(results)

        # End DB session
        if self.db and self.session_id:
            try:
                self.db.end_session(self.session_id)
            except Exception as e:
                logger.warning(f"Error: {e}")

        try:
            shutil.rmtree("temp/question_audio")
        except Exception:
            pass



    def _compile_results(self):
        """Compile all test results into output format."""
        spiral_data  = getattr(self.modules["Spiral"], 'results', {})
        trail_data   = getattr(self.modules["Trail"],  'results', {})
        shape_data   = getattr(self.modules["Shapes"], 'results', {})
        memory_data  = getattr(self.modules["Memory"], 'results', {})

        user_id = self.user.id if self.user else None

        return {
            "consult_id": self.id,
            "user_id": str(user_id) if user_id else None,
            "consult_time": self.date.strftime("%Y-%m-%d"),
            "consult_data": {
                "shape": shape_data,
                "spiral": spiral_data,
                "memory": memory_data,
                "trail": trail_data,
            }
        }

    def _save_results_local(self, results):
        """Save consultation results to local JSON."""
        self.output = results

        if not self.user:
            return

        base_path = "data"
        record_path = os.path.join(base_path, "consult_records")
        os.makedirs(record_path, exist_ok=True)

        user_path = os.path.join(record_path, f"user_{self.user.id}")
        os.makedirs(user_path, exist_ok=True)

        consult_path = os.path.join(user_path, f"consult_{self.id}.json")

        with open(consult_path, "w") as f:
            json.dump(self.output, f, cls=NpEncoder, indent=4)



    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def loop(self, infinite=False):
        """
        Main consultation loop.

        Args:
            infinite: Loop through tests indefinitely (for demos)
        """
        self.entry_sequence()

        while self.running:
            if self.seamless:
                for game_number, module_name in enumerate(self.module_order):
                    started_at = self.now()

                    # Log game_start event
                    self._log_game_event(module_name, game_number, 'game_start', started_at)

                    self.modules[module_name].loop()

                    completed_at = self.now()

                    # Create DB session right after login
                    if module_name == "Login" and self.user:
                        self._start_db_session()

                    # Log game_end event
                    self._log_game_event(module_name, game_number, 'game_end', completed_at)

                    # Save game result to DB
                    self._save_game_to_db(module_name, game_number, started_at, completed_at)

                    self.update_display()

                self.running = False

            else:
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN:
                        if event.key == pg.K_ESCAPE:
                            self.running = False
                        elif event.key == pg.K_z and (pg.key.get_mods() & pg.KMOD_CTRL):
                            import sys
                            pg.quit()
                            sys.exit(0)
                        elif event.key == pg.K_s:
                            self.take_screenshot()

                    elif event.type == pg.MOUSEBUTTONDOWN:
                        button_id = self.touch_screen.click_test(
                            self.get_relative_mose_pos()
                        )

                        if button_id == 1:  # Start button
                            self.touch_screen.kill_sprites()
                            self.update_display()

                            module_name = self.module_order[self.module_idx]
                            started_at = self.now()

                            module = self.modules[module_name]
                            module.running = True
                            module.loop()

                            completed_at = self.now()

                            if module_name == "Login" and self.user:
                                self._start_db_session()

                            self._save_game_to_db(module_name, self.module_idx, started_at, completed_at)

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
    os.environ['SDL_VIDEO_WINDOW_POS'] = "1029,600"  # Top physical screen (HDMI-A-2)
    pg.init()
    pg.font.init()
    pg.event.pump()

    consult = Consultation(
        pi=False,
        authenticate=False,
        seamless=True,
        local=True,
        enable_speech=False,
        scale=0.8,
    )

    consult.loop()
    
