"""
Shape Searcher — visual perception and working memory assessment game.

Patients compare sets of coloured shapes displayed on the touchscreen,
pressing 'Same' or 'Different' to record their judgement. Three question
types of increasing difficulty are used: perception (both sets visible),
shape binding (working memory, shape only), and colour binding
(working memory, shape + colour).
"""

import random
import time

import numpy as np
import pandas as pd
import pygame as pg

from hero.consultation.display_screen import DisplayScreen
from hero.consultation.screen import Colours
from hero.consultation.touch_screen import TouchScreen, GameObjects, GameButton
from hero.consultation.utils import take_screenshot, Buttons, ButtonModule

rocket = [(4, 0), (8, 4), (5, 8), (4, 5), (3, 8), (0, 4)]
lightning_1 = [(5, 0), (5, 3), (8, 3), (3, 8), (3, 5), (0, 5)]
lightning_2 = [(4, 0), (4, 2.5), (8, 5.5), (4, 8), (4, 5.5), (0, 2.5)]
arrow_1 = [(1.5, 0), (4, 0), (8, 4), (3.25, 8), (0, 8), (5, 3.5)]
arrow_2 = [(0, 0), (7, 5), (7, 8), (3, 5), (3, 8), (0, 8)]
gem = [(5, 0), (8, 2), (8, 5), (3, 8), (0, 6), (0, 3)]
staple = [(1, 0), (7, 0), (7, 8), (1, 8), (3.5, 5.5), (3.5, 2.5)]

shapes = {"rocket": rocket, "lightning_1": lightning_1,
          "arrow_1": arrow_1, "lightning_2": lightning_2,
          "arrow_2": arrow_2, "gem": gem, "staple": staple}

shape_colours = [Colours.red, Colours.blue, Colours.green, Colours.yellow, Colours.shadow]


class Circle(pg.sprite.Sprite):
    """
    A circular target sprite used in the speed sub-task.

    Attributes:
        object_type: Always 'circle', used by GameObjects.draw() for dispatch.
        image: Pygame Surface with the filled circle.
        rect: pg.Rect defining the sprite's position and bounding box.
    """

    def __init__(self, position, size, colour):
        """
        Initialise a Circle sprite.

        Args:
            position: (x, y) top-left position of the bounding square.
            size: Diameter of the circle in pixels.
            colour: Colours enum value for the circle fill.
        """
        super().__init__()
        self.object_type = "circle"
        surf_size = pg.Vector2(size, size)
        self.image = pg.Surface(surf_size, pg.SRCALPHA)
        pg.draw.circle(self.image, colour.value, surf_size / 2, size / 2)
        self.rect = pg.Rect(position, surf_size)

    def is_clicked(self, pos):
        """
        Test whether a position falls within the bounding rectangle.

        Args:
            pos: (x, y) position to test.

        Returns:
            True if pos is within the bounding rect, False otherwise.
        """
        if self.rect.collidepoint(pos):
            return True
        else:
            return False

    def click_return(self):
        """
        Return value when the circle is clicked.

        Returns:
            True, indicating the circle was tapped.
        """
        return True


def create_shape_surf(name, scale, colour):
    """
    Render a named shape polygon to a Surface.

    Args:
        name: Key into the shapes dict (e.g. 'rocket', 'gem').
        scale: Pixel scale factor applied to the 8×8 unit shape grid.
        colour: Colours enum value for the polygon fill.

    Returns:
        pg.Surface containing the rendered shape.
    """
    shape_surf = pg.Surface((8 * scale, 8 * scale), pg.SRCALPHA)
    shape_coords = [(coord[0] * scale, coord[1] * scale) for coord in shapes[name]]
    pg.draw.polygon(shape_surf, colour.value, shape_coords)
    return shape_surf


class ShapeSearcher:
    """
    Visual shape matching game for the HERO consultation system.

    Presents three sub-tasks in sequence: perception (shapes always visible),
    shape binding (working memory without colour), and colour binding (working
    memory with colour). Patients press Same or Different for each question.

    Attributes:
        parent: Parent Consultation instance, or None in standalone mode.
        display_size: Vector2 dimensions of the display area.
        display_screen: DisplayScreen for the upper screen.
        touch_screen: TouchScreen for the lower screen.
        button_module: ButtonModule for hardware button input.
        same_button: GameButton for the 'Same' response.
        different_button: GameButton for the 'Different' response.
        shape_size: Vector2 size used for overlap-check bounding boxes.
        match: 1 if the current question is a match, 0 if not.
        turns: Total number of questions answered so far.
        scores: List of correct counts per question type [perception, shape, colour].
        question_counts: Target question counts per type [perception, shape, colour].
        answer_times: List of per-question reaction times in seconds.
        trial_log: List of per-trial dicts with question and response data.
        start_time: monotonic timestamp of when the current question was displayed.
        results: Dict populated by exit_sequence with summary statistics.
        running: Main loop control flag.
        auto_run: If True, simulate responses automatically.
        show_info: Whether the info overlay is currently visible.
        power_off: If True, show the power-off splash screen.
    """

    def __init__(self, size=(1024, 600), parent=None, auto_run=False):
        """
        Initialise the Shape Searcher and set up button layout.

        Args:
            size: Tuple (width, height) used in standalone mode without a parent.
            parent: Parent Consultation instance. If provided, screens are shared.
            auto_run: If True, simulate responses automatically (for testing).
        """
        self.parent = parent
        if parent is not None:
            self.display_size = parent.display_size
            self.bottom_screen = parent.bottom_screen
            self.top_screen = parent.top_screen
            self.display_screen = DisplayScreen(self.display_size, avatar=parent.avatar)
            self.button_module = parent.button_module

        else:
            self.display_size = pg.Vector2(size)
            self.window = pg.display.set_mode((self.display_size.x, self.display_size.y * 2), pg.SRCALPHA)

            self.top_screen = self.window.subsurface(((0, 0), self.display_size))
            self.bottom_screen = self.window.subsurface((0, self.display_size.y), self.display_size)
            self.display_screen = DisplayScreen(self.display_size)
            self.button_module = ButtonModule(pi=False)

        self.display_screen.instruction = None

        self.touch_screen = TouchScreen(self.display_size)
        self.button_size = pg.Vector2(self.display_size.x * 0.45, 100)
        button_pad = pg.Vector2(self.touch_screen.size.y, self.touch_screen.size.y) * 0.05
        self.same_button = GameButton(position=(button_pad.x,
                                                self.touch_screen.size.y - button_pad.y - self.button_size.y),
                                      size=self.button_size, id=1, text="Same")
        self.different_button = GameButton(position=(self.touch_screen.size.x - button_pad.x - self.button_size.x,
                                                     self.touch_screen.size.y - button_pad.y - self.button_size.y),
                                           size=self.button_size, id=0, text="Different")

        self.shape_size = pg.Vector2(80, 80)
        self.match = None
        self.turns = 0
        self.scores = [0, 0, 0]
        self.question_counts = [10, 0, 0]
        self.answer_times = []
        self.trial_log = []
        self.start_time = None
        self.results = {}

        self.question_types = (["perception" for _ in range(self.question_counts[0])] +
                               ["shape" for _ in range(self.question_counts[1])] +
                               ["colour" for _ in range(self.question_counts[2])])

        self.running = False
        self.auto_run = auto_run
        self.show_info = False
        self.power_off = False

    def instruction_loop(self, question):
        """
        Show an instruction screen for the given question type and wait for Start.

        No-ops in auto_run mode.

        Args:
            question: Question type string — 'perception', 'shape', or 'colour'.
        """
        if self.auto_run:
            return

        temp_instruction = self.display_screen.instruction
        self.display_screen.state = 1
        self.display_screen.refresh()
        self.display_screen.instruction = None

        button_rect = pg.Rect((self.display_size - pg.Vector2(300, 200))/2, (300, 200))
        start_button = GameButton(position=button_rect.topleft, size=button_rect.size, text="START", id=1)
        self.touch_screen.sprites = GameObjects([start_button])
        info_rect = pg.Rect(0.3 * self.display_size.x, 0, 0.7 * self.display_size.x, 0.8 * self.display_size.y)
        pg.draw.rect(self.display_screen.surface, Colours.white.value, info_rect)

        self.display_screen.add_multiline_text("Match the Shapes!", rect=info_rect.scale_by(0.9, 0.9), font_size=50)

        if question == "perception":
            info_text = (
                "You will see three coloured shapes located above and below a black line. " +
                "Your task is to say whether the shapes that you see above the line are the same as the shapes below the line.")
            self.display_screen.add_multiline_text(
                rect=info_rect.scale_by(0.9, 0.9), text=info_text, center_vertical=True)
        else:
            info_text = ("You will now have to try and remember a set of coloured shapes. You will be shown two or three " +
                "shapes, which will then disappear after a short amount of time. A second set of shapes will then appear, " +
                "and your task is to identify if the two sets are the same or different. Sets are considered the " +
                "same if each shape and colour matches.")
            self.display_screen.add_multiline_text(
                rect=info_rect.scale_by(0.9, 0.9), text=info_text, center_vertical=True)

        self.update_display()
        if self.parent:
            self.parent.speak_text(
                info_text, visual=True, display_screen=self.display_screen, touch_screen=self.touch_screen)

        self.update_display()
        wait = True
        while wait:
            for event in pg.event.get():
                if event.type == pg.MOUSEBUTTONDOWN and not self.power_off:
                    pos = pg.Vector2(pg.mouse.get_pos()) - pg.Vector2(0, self.display_size.y)
                    selection = self.touch_screen.click_test(pos)
                    if selection is not None:
                        wait = False

                elif event.type == pg.KEYDOWN and not self.power_off:
                    if event.key == pg.K_s:
                        if self.parent:
                            take_screenshot(self.parent.window)
                        else:
                            take_screenshot(self.window, "Shape_match")

            selected = self.button_module.check_pressed()
            if selected == Buttons.power:
                self.power_off = not self.power_off
                self.display_screen.power_off = self.power_off
                self.touch_screen.power_off = self.power_off
                if self.power_off:
                    self.display_screen.instruction = None
                self.update_display()

        self.touch_screen.kill_sprites()
        self.display_screen.state = 0
        self.display_screen.refresh()
        self.display_screen.instruction = temp_instruction

        self.update_display()

    def update_display(self):
        """Blit both screens to the physical displays."""
        self.top_screen.blit(self.display_screen.get_surface(), (0, 0))
        self.bottom_screen.blit(self.touch_screen.get_surface(), (0, 0))
        pg.display.flip()

    def entry_sequence(self):
        """Speak the intro and start the first perception question block."""
        self.update_display()
        if self.parent:
            self.parent.speak_text("your next set of tasks will all involve matching sets of shapes",
                                   display_screen=self.display_screen, touch_screen=self.touch_screen)

        self.display_screen.instruction = "Match the sets!"
        self.instruction_loop(question="perception")
        self.perception_question()
        self.display_screen.refresh()
        self.touch_screen.sprites = GameObjects([self.same_button, self.different_button])
        self.update_display()
        self.running = True
        self.start_time = time.monotonic()

    def check_ok(self, pos, existing):
        """
        Test whether a candidate position overlaps any existing shape position.

        Args:
            pos: Candidate (x, y) position for the new shape.
            existing: List of existing positions to check against.

        Returns:
            True if pos does not collide with any existing position.
        """
        for point in existing:
            invalid = pg.Rect((point.x - self.shape_size.x, point.y - self.shape_size.y),
                              self.shape_size * 2).scale_by(1.1, 1.1)
            if invalid.collidepoint(pos):
                return False
        else:
            return True

    def generate_symbol_set(self, area, shape_count, scene_shapes=None, colours=None):
        """
        Generate a random set of shapes placed within the given area.

        If scene_shapes is provided and self.match is False, a completely
        different set of shapes is selected to ensure a non-matching pair.

        Args:
            area: pg.Rect defining the placement region.
            shape_count: Number of shapes to generate.
            scene_shapes: Optional Series of shape names to reuse or replace.
            colours: Optional Series of Colours enum values to reuse.

        Returns:
            Tuple of (symbol_list, scene_shapes, colours) where symbol_list is
            a list of (Surface, position) pairs.
        """
        area = area.copy()
        area.size -= self.shape_size
        if colours is None:
            colours = pd.Series(shape_colours)[np.random.permutation(len(shape_colours))[range(shape_count)]]
        if scene_shapes is None:
            scene_shapes = pd.Series(shapes.keys())[np.random.permutation(len(shapes))[range(shape_count)]]
        elif not self.match:
            new_shapes = pd.Series(shapes.keys())[np.random.permutation(len(shapes))[range(shape_count)]]
            while all(x in new_shapes for x in scene_shapes):
                new_shapes = pd.Series(shapes.keys())[np.random.permutation(len(shapes))[range(shape_count)]]
            scene_shapes = new_shapes

        positions = []
        for idx in range(shape_count):
            test_pos = (area.topleft +
                        pg.Vector2(np.random.randint(0, area.width + 1),
                                   np.random.randint(0, area.height + 1)))

            while not self.check_ok(test_pos, positions):
                test_pos = (area.topleft +
                            pg.Vector2(np.random.randint(0, area.width + 1),
                                       np.random.randint(0, area.height + 1)))

            positions.append(test_pos)

        return [(create_shape_surf(name, 10, colour), position) for
                name, colour, position in zip(scene_shapes, colours, positions)], scene_shapes, colours

    def perception_question(self):
        """
        Display two sets of three shapes split by a horizontal divider.

        Both sets are always visible simultaneously. self.match determines
        whether the sets are identical (colour and shape) or differ.
        """
        self.touch_screen.refresh()

        place_area = pg.Rect((0, 0), self.touch_screen.size).scale_by(0.95, 0.7)
        place_area.topleft -= pg.Vector2(0, self.touch_screen.size.y * 0.1)

        place_area_top = pg.Rect(place_area.topleft, (place_area.width, place_area.height / 2)).scale_by(0.9, 0.9)
        place_area_bottom = pg.Rect(place_area.midleft, (place_area.width, place_area.height / 2)).scale_by(0.9, 0.9)

        symbol_set_1, scene_shapes, scene_colours = self.generate_symbol_set(place_area_top, 3)
        self.match = np.random.randint(0, 2)
        symbol_set_2, _, _ = self.generate_symbol_set(place_area_bottom, 3, colours=scene_colours,
                                                      scene_shapes=scene_shapes)

        pg.draw.line(self.touch_screen.surface, Colours.black.value, place_area.midleft, place_area.midright, width=5)

        for symbol, pos in symbol_set_1:
            self.touch_screen.add_surf(symbol, pos)
        for symbol, pos in symbol_set_2:
            self.touch_screen.add_surf(symbol, pos)

        self.update_display()

    def binding_question(self, colour=True):
        """
        Display a working memory shape matching question.

        Shows the first set briefly, clears the screen, then shows the second
        set for the patient to compare from memory.

        Args:
            colour: If True, shapes are coloured (colour binding). If False,
                all shapes use the same colour (shape binding only).
        """
        self.touch_screen.refresh()
        self.display_screen.instruction = "Do the sets match?"
        self.touch_screen.kill_sprites()
        place_area = pg.Rect((0, 0), self.touch_screen.size).scale_by(0.7, 0.7)
        place_area.topleft -= pg.Vector2(0, self.touch_screen.size.y * 0.1)

        shape_count = np.random.randint(2, 4)

        if not colour:
            colours = [Colours.hero_blue for _ in range(shape_count)]
        else:
            colours = None

        symbol_set_1, scene_shapes, scene_colours = self.generate_symbol_set(place_area, shape_count, colours=colours)
        self.match = np.random.randint(0, 2)
        symbol_set_2, _, _ = self.generate_symbol_set(place_area, shape_count, colours=scene_colours,
                                                      scene_shapes=scene_shapes)

        for symbol, pos in symbol_set_1:
            self.touch_screen.add_surf(symbol, pos)

        self.update_display()
        if not self.auto_run:
            time.sleep(2)

        self.touch_screen.refresh()
        self.update_display()
        if not self.auto_run:
            time.sleep(2)

        for symbol, pos in symbol_set_2:
            self.touch_screen.add_surf(symbol, pos)
        self.touch_screen.sprites = GameObjects([self.same_button, self.different_button])
        self.update_display()

    def speed_question(self):
        """Display a random red circle and prompt the patient to tap it."""
        self.touch_screen.refresh()
        self.touch_screen.kill_sprites()
        self.display_screen.instruction = "Touch the dot!"

        size = np.random.randint(20, 51)
        place_area = pg.Rect((0, 0), self.touch_screen.size).scale_by(0.6, 0.6)
        place_area.size -= pg.Vector2(size, size)

        pos = (place_area.topleft +
               pg.Vector2(np.random.randint(0, place_area.width + 1),
                          np.random.randint(0, place_area.height + 1)))

        circle = Circle(pos, size, Colours.red)
        self.touch_screen.sprites = GameObjects([circle])
        self.update_display()

    def exit_sequence(self):
        """
        Compile per-question results into self.results.

        In auto_run mode, generates synthetic reaction times before compiling.
        """
        if self.auto_run:
            self.answer_times = [random.gauss(mu=1, sigma=0.1) for _ in range(self.turns)]

        total_q = self.turns
        correct_q = self.scores[0]  # only perception mode runs (question_counts = [10, 0, 0])
        self.results = {
            'total_questions': total_q,
            'correct': correct_q,
            'incorrect': total_q - correct_q,
            'accuracy_percent': round((correct_q / total_q) * 100, 1) if total_q > 0 else 0,
            'avg_reaction_time_s': round(sum(self.answer_times) / len(self.answer_times), 4) if self.answer_times else None,
            'trial_log': self.trial_log,
        }

    def process_selection(self, selection):
        """
        Record a patient response and advance to the next question.

        Updates scores, trial log, and turn count. Transitions between
        question types when a block is complete, and ends the game if the
        patient scores below 80% on the perception block.

        Args:
            selection: 1 if the patient pressed 'Same', 0 for 'Different'.
        """
        reaction_time = time.monotonic() - self.start_time
        self.answer_times.append(reaction_time)

        correct = (selection == self.match)
        if correct:
            if self.question_types[self.turns] == "perception":
                self.scores[0] += 1
            elif self.question_types[self.turns] == "shape":
                self.scores[1] += 1
            else:
                self.scores[2] += 1

        self.trial_log.append({
            'question_num': self.turns + 1,
            'correct': bool(correct),
            'reaction_time_s': round(reaction_time, 4),
            'patient_said_same': bool(selection == 1),
            'was_same': bool(self.match == 1),
        })

        self.turns += 1
        if self.turns == sum(self.question_counts):
            self.running = False
        elif self.turns == self.question_counts[0] and self.scores[0] < 0.8 * self.question_counts[0]:
            # Early exit if perception accuracy is too low to proceed
            self.running = False
        else:
            if self.turns == self.question_counts[0]:
                if self.parent:
                    self.display_screen.refresh()
                    self.touch_screen.kill_sprites()
                    self.touch_screen.refresh()
                    self.display_screen.instruction = None
                    self.update_display()

                    self.parent.speak_text("Well done, we are now moving onto the second task",
                                           display_screen=self.display_screen, touch_screen=self.touch_screen)
                    self.display_screen.instruction = "Do the sets match?"

                self.touch_screen.refresh()
                self.instruction_loop("shape")
                self.display_screen.refresh()

            if self.question_types[self.turns] == "perception":
                self.perception_question()
            elif self.question_types[self.turns] == "shape":
                self.binding_question(colour=False)
            else:
                self.binding_question()

        self.start_time = time.monotonic()

    def button_actions(self, selected):
        """
        Handle hardware button presses during the game.

        Args:
            selected: Buttons enum member for the pressed button.
        """
        if selected == Buttons.info and not self.power_off:
            self.show_info = not self.show_info
            self.toggle_info_screen()
        elif selected == Buttons.power:
            self.power_off = not self.power_off
            self.display_screen.power_off = self.power_off
            self.touch_screen.power_off = self.power_off
            if self.power_off:
                self.display_screen.instruction = None
                self.update_display()
            else:
                self.toggle_info_screen()
        else:
            ...

    def toggle_info_screen(self):
        """Toggle the instruction overlay on the upper display."""
        if self.show_info:
            self.display_screen.state = 1
            self.display_screen.instruction = None

            info_rect = pg.Rect(0.3 * self.display_size.x, 0, 0.7 * self.display_size.x, 0.8 * self.display_size.y)
            pg.draw.rect(self.display_screen.surface, Colours.white.value, info_rect)

            self.display_screen.add_multiline_text("Match the shapes!", rect=info_rect.scale_by(0.9, 0.9),
                                                   font_size=50)

            if self.turns < self.question_counts[0]:
                info_text = (
                    "Are the shapes above the black line the same as the ones below? " +
                    "Each shape must be the same colour for sets to match.")
                self.display_screen.add_multiline_text(
                    rect=info_rect.scale_by(0.9, 0.9), text=info_text,
                    center_vertical=True, font_size=40)
            else:
                info_text = (
                    "You will now have to try and remember a set of coloured shapes. You will be shown two or three " +
                    "shapes, which will then disappear after a short amount of time. A second set of shapes will then appear, " +
                    "and your task is to identify if the two sets are the same or different. Sets are considered the " +
                    "same if each shape and colour matches.")
                self.display_screen.add_multiline_text(
                    rect=info_rect.scale_by(0.9, 0.9), text=info_text, center_vertical=True)

            self.update_display()
        else:
            self.display_screen.refresh()
            self.display_screen.state = 0
            self.display_screen.instruction = "Do the sets match?"
            self.update_display()

    def loop(self):
        """
        Main event loop — called by the orchestrator.

        In auto_run mode, simulates responses with weighted random choices.
        In normal mode, handles touch input and hardware button presses.
        """
        self.entry_sequence()
        while self.running:
            if self.auto_run:
                if self.question_types[self.turns] == "perception":
                    weights = [9, 1]
                elif self.question_types[self.turns] == "shape":
                    weights = [65, 35]
                else:
                    weights = [8, 2]

                if self.match:
                    weights.reverse()

                selection = random.choices([0, 1], weights=weights, k=1)[0]
                self.process_selection(selection)
            else:
                for event in pg.event.get():
                    if event.type == pg.KEYDOWN and not self.power_off:
                        if event.key == pg.K_s:
                            if self.parent:
                                take_screenshot(self.parent.window)
                            else:
                                take_screenshot(self.window, "Shape_match")
                        elif event.key == pg.K_ESCAPE:
                            self.running = False

                    elif event.type == pg.MOUSEBUTTONDOWN and not self.power_off:
                        pos = pg.Vector2(pg.mouse.get_pos()) - pg.Vector2(0, self.display_size.y)
                        selection = self.touch_screen.click_test(pos)
                        if selection is not None:
                            self.process_selection(selection)
                        pg.event.clear()

                    elif event.type == pg.QUIT:
                        self.running = False

                selected = self.button_module.check_pressed()
                if selected is not None:
                    self.button_actions(selected)

        self.exit_sequence()
