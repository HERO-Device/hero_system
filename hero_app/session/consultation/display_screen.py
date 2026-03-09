"""
Top-screen display surface for the HERO consultation system.

Renders the avatar, speech bubble, instruction bar, and background
onto the upper physical screen during patient consultations.
"""

import pygame as pg

from consultation.avatar import Avatar
from consultation.screen import Screen, BlitLocation, Colours


class DisplayScreen(Screen):
    """
    Upper consultation screen showing the avatar and session instructions.

    Extends Screen with a fixed layout: HERO background and logo on the base
    layer, a hero_blue instruction bar at the bottom, and an avatar that
    repositions based on state. A speech bubble is rendered when speech_text
    is set. A power-off surface replaces the display when power_off is True.

    Attributes:
        avatar: Avatar instance used for mouth animation during speech.
        instruction: Text string rendered in the info bar. None hides the bar.
        speech_text: Text rendered in the speech bubble. None hides the bubble.
        speech_textbox: pg.Rect defining the speech bubble area.
        info_textbox: pg.Rect defining the bottom instruction bar area.
        state: Avatar position mode (0 = centre, 1 = left).
        power_off: If True, get_surface() returns the power-off splash screen.
    """

    def __init__(self, size, info_height=0.2, avatar=None):
        """
        Initialise the display screen and bake the static background into the base layer.

        Args:
            size: Tuple or Vector2 (width, height) of the display area.
            info_height: Fraction of screen height reserved for the instruction bar.
            avatar: Optional Avatar instance. A new one is created if not provided.
        """
        super().__init__(size, colour=Colours.white)
        if avatar:
            self.avatar = avatar
        else:
            self.avatar = Avatar()

        self.instruction = None
        self.speech_text = None
        self.speech_textbox = pg.Rect(0.44*self.size.x, 0.05*self.size.y, 0.5*self.size.x, 0.65*self.size.y)
        self.info_textbox = pg.Rect(0, (1-info_height)*self.size.y, self.size.x, info_height*self.size.y)
        self.load_image("hero/consultation/resources/graphics/backgrounds/background.png", fill=True, base=True, pos=(0, 0))
        self.load_image("hero/consultation/resources/graphics/logo.png", size=(38*3, 53*600/256), pos=(262*3, 52*600/256), base=True)
        pg.draw.rect(self.base_surface, Colours.hero_blue.value, self.info_textbox)

        self.power_off_surface = pg.Surface((self.size.x, self.size.y), pg.SRCALPHA)
        self.power_off_surface.fill(Colours.white.value)

        image = pg.image.load("hero/consultation/resources/graphics/hero_text.png")
        pos = self.size/2

        imageRect = pg.Rect(pos, image.get_size())
        imageRect.topleft = pg.Vector2(imageRect.topleft) - pg.Vector2(imageRect.size) / 2

        self.power_off_surface.blit(image, imageRect)
        self.state = 0
        self.power_off = False

    def get_surface(self):
        """
        Composite the display into a single surface for blitting to the screen.

        Renders the instruction bar, avatar, and speech bubble onto the dynamic
        layer before compositing base + surface + sprite layers.

        Returns:
            pg.Surface ready to blit to the top physical screen.
        """
        if self.power_off:
            return self.power_off_surface

        if self.instruction:
            self.add_multiline_text(self.instruction, self.info_textbox, center_horizontal=True, center_vertical=True, colour=Colours.white, font_size=50)

        if self.speech_text:
            self.add_surf(self.avatar.get_surface(), (0, self.size.y - self.info_textbox.h),
                          location=BlitLocation.bottomLeft)

            border = 5
            triangle_points = (self.speech_textbox.topleft + pg.Vector2(0, 0.8*self.speech_textbox.h),
                               self.speech_textbox.topleft + pg.Vector2(0, 0.9*self.speech_textbox.h),
                               self.speech_textbox.topleft + pg.Vector2(-0.1*self.speech_textbox.w, 0.9*self.speech_textbox.h))

            self.add_multiline_text(self.speech_text, self.speech_textbox.inflate(-2*border, -2*border),
                                    center_vertical=True, center_horizontal=True, bg_colour=Colours.white, font_size=50,
                                    colour=Colours.hero_blue)
            pg.draw.rect(self.surface, Colours.lightGrey.value, self.speech_textbox, border_radius=int(border * 2),
                         width=border)
            pg.draw.polygon(self.surface, Colours.lightGrey.value, triangle_points)

        elif self.state == 0:
            self.add_surf(self.avatar.get_surface(), (self.size.x/2, self.size.y - self.info_textbox.h), location=BlitLocation.midBottom)
        elif self.state == 1:
            self.add_surf(self.avatar.get_surface(), (-self.size.x*0.05, self.size.y - self.info_textbox.h), location=BlitLocation.bottomLeft)

        display_surf = self.base_surface.copy()
        display_surf.blit(self.surface, (0, 0))
        display_surf.blit(self.sprite_surface, (0, 0))

        return display_surf
