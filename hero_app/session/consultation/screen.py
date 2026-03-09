"""
Pygame surface primitives for the HERO consultation UI.

Provides the base Screen class used by all display layers, along with
colour, font, and blit-location enumerations shared across the system.
"""

import math
from enum import Enum

import pygame as pg
import pygame.freetype


class Colours(Enum):
    """Named colour palette used across all HERO UI components."""

    clear = pg.SRCALPHA
    white = pg.Color(255, 255, 255)
    black = pg.Color(1, 1, 1)
    darkGrey = pg.Color(60, 60, 60)
    midGrey = pg.Color(150, 150, 150)
    lightGrey = pg.Color(200, 200, 200)
    green = pg.Color(69, 181, 67)
    red = pg.Color(181, 67, 67)
    shadow = pg.Color(180, 180, 180)
    blue = pg.Color(67, 113, 181)
    yellow = pg.Color(252, 198, 3)
    hero_blue = pg.Color("#274251")
    light_blue = pg.Color("#4f86a5")


class BlitLocation(Enum):
    """Anchor point enum for positioning surfaces and text during blit operations."""

    topLeft = 0
    midTop = 1
    topRight = 2
    bottomLeft = 3
    midBottom = 4
    bottomRight = 5
    midLeft = 6
    midRight = 7
    centre = 8


class Fonts:
    """
    Font presets loaded from the HERO Calibri resource file.

    Attributes:
        large: 50pt font.
        normal: 30pt font.
        small: 15pt font.
        custom: Dynamically sized font, defaults to normal.
    """

    def __init__(self):
        self.large = pg.font.Font("hero/consultation/resources/fonts/calibri-regular.ttf", 50)
        self.normal = pg.font.Font("hero/consultation/resources/fonts/calibri-regular.ttf", 30)
        self.small = pg.font.Font("hero/consultation/resources/fonts/calibri-regular.ttf", 15)
        self.custom = self.normal

    def update_custom(self, size):
        """
        Set the custom font to a specific point size.

        Args:
            size: Point size for the custom font.
        """
        self.custom = pg.font.Font("hero/consultation/resources/fonts/calibri-regular.ttf", size=size)


class Screen:
    """
    Layered pygame surface wrapper used as the base for all HERO screens.

    Manages three composited layers — base (static background), surface
    (dynamic content), and sprite (interactive elements) — which are
    flattened into a single surface via get_surface().

    Attributes:
        size: Surface dimensions as a Vector2.
        base_surface: Static background layer.
        surface: Dynamic content layer, cleared on refresh().
        sprite_surface: Interactive sprite layer, cleared on refresh().
        font: Active font used for text rendering.
        fonts: Fonts instance providing size presets.
        colour: Background fill colour.
    """

    def __init__(self, size, font=None, colour=None):
        """
        Initialise the Screen with the given dimensions and optional defaults.

        Args:
            size: Tuple or Vector2 (width, height) of the surface.
            font: Optional pygame Font to use as the active font. Defaults to Fonts.normal.
            colour: Optional Colours enum or pg.Color to fill the base surface.
        """
        self.size = pg.Vector2(size)
        self.base_surface = pg.Surface(size, pg.SRCALPHA)
        self.surface = pg.Surface(size, pg.SRCALPHA)
        self.sprite_surface = pg.Surface(size, pg.SRCALPHA)
        if font:
            self.fonts = Fonts()
            self.font: pg.font.Font = font
        else:
            self.fonts = Fonts()
            self.font = self.fonts.normal

        if colour:
            if not type(colour) == pg.Color:
                colour = colour.value

        self.colour = colour
        if colour:
            self.base_surface.fill(colour)

    def add_surf(self, surf: pg.Surface, pos=(0, 0), base=False, location=BlitLocation.topLeft, sprite=False):
        """
        Blit a surface onto one of the screen layers.

        Args:
            surf: Surface to blit.
            pos: (x, y) position to blit at.
            base: If True, blit onto the base layer instead of the dynamic layer.
            location: BlitLocation anchor point applied to pos.
            sprite: If True, blit onto the sprite layer.
        """
        surf_rect = pg.Rect(pos, surf.get_size())

        if location == BlitLocation.centre:
            surf_rect.topleft -= pg.Vector2(surf_rect.size) / 2
        elif location == BlitLocation.topRight:
            surf_rect.x -= surf_rect.width
        elif location == BlitLocation.bottomLeft:
            surf_rect.y -= surf_rect.height
        elif location == BlitLocation.midBottom:
            surf_rect.y -= surf_rect.height
            surf_rect.x -= surf_rect.width / 2

        if sprite:
            self.sprite_surface.blit(surf, surf_rect.topleft)
        elif base:
            self.base_surface.blit(surf, surf_rect.topleft)
        else:
            self.surface.blit(surf, surf_rect.topleft)

    def load_image(self, path, pos=(0, 0), fill=False, base=False, size=None, scale=None, location=BlitLocation.topLeft):
        """
        Load an image from disk and blit it onto the screen.

        Args:
            path: File path to the image.
            pos: (x, y) position to blit at.
            fill: If True, scale the image to fill the entire surface.
            base: If True, blit onto the base layer.
            size: Optional (width, height) to scale the image to.
            scale: Optional Vector2 scale factor applied to the image dimensions.
            location: BlitLocation anchor point applied to pos.
        """
        image = pg.image.load(path)

        if size:
            image = pg.transform.scale(image, size)
        elif scale:
            image = pg.transform.scale(image, (image.get_size()[0] * scale.x, image.get_size()[1] * scale.y))
        elif fill:
            image = pg.transform.scale(image, self.size)

        imageRect = pg.Rect(pos, image.get_size())

        if location == BlitLocation.centre:
            imageRect.topleft = pg.Vector2(imageRect.topleft) - pg.Vector2(imageRect.size) / 2
        elif location == BlitLocation.topRight:
            imageRect.x -= imageRect.width

        if base:
            self.base_surface.blit(image, imageRect.topleft)
        else:
            self.surface.blit(image, imageRect.topleft)

    def add_image(self, image, pos=pg.Vector2(0, 0), fill=False, scale=None, size=None, location=BlitLocation.topLeft, base=False):
        """
        Blit an already-loaded image surface onto the screen.

        Args:
            image: pygame Surface to blit.
            pos: (x, y) position to blit at.
            fill: If True, scale the image to fill the entire surface.
            scale: Optional Vector2 scale factor applied to image dimensions.
            size: Optional (width, height) to scale the image to.
            location: BlitLocation anchor point applied to pos.
            base: If True, blit onto the base layer.
        """
        if base:
            surf = self.base_surface
        else:
            surf = self.surface

        if size:
            image = pg.transform.scale(image, size)
        elif scale:
            image = pg.transform.scale(image, (image.get_size()[0] * scale.x, image.get_size()[1] * scale.y))
        elif fill:
            image = pg.transform.scale(image, self.size)

        size = pg.Vector2(image.get_size())
        if location == BlitLocation.centre:
            surf.blit(image, pos - size / 2)
        elif location == BlitLocation.midBottom:
            newPos = pg.Vector2(pos.x - (size.x / 2), pos.y - size.y)
            surf.blit(image, newPos)
        else:
            surf.blit(image, pos)

    def add_text(self, text, pos, lines=1, colour=Colours.black, bg_colour=None, location=BlitLocation.topLeft, sprite=False, base=False):
        """
        Render a single line of text onto the screen.

        Args:
            text: String to render.
            pos: (x, y) position to blit at.
            lines: Unused; reserved for future multi-line support.
            colour: Colours enum value for the text colour.
            bg_colour: Optional Colours enum value for a background fill behind the text.
            location: BlitLocation anchor point applied to pos.
            sprite: If True, render onto the sprite layer.
            base: If True, render onto the base layer.
        """
        text_surf = self.font.render(text, True, colour.value)
        if bg_colour:
            bg_surf = pg.Surface(text_surf.get_size(), pg.SRCALPHA)
            bg_surf.fill(bg_colour.value)
            bg_surf.blit(text_surf, (0, 0))
            text_surf = bg_surf

        blitPos = pos
        size = pg.Vector2(text_surf.get_size())
        if location == BlitLocation.centre:
            blitPos -= size / 2
        elif location == BlitLocation.topRight:
            blitPos -= pg.Vector2(size.x, 0)
        elif location == BlitLocation.midTop:
            blitPos -= pg.Vector2(size.x / 2, 0)
        elif location == BlitLocation.midBottom:
            blitPos -= pg.Vector2(size.x / 2, size.y)

        if sprite:
            self.sprite_surface.blit(text_surf, blitPos)
        elif base:
            self.base_surface.blit(text_surf, blitPos)
        else:
            self.surface.blit(text_surf, blitPos)

    def add_multiline_text(self, text, rect, location=BlitLocation.topLeft, center_horizontal=False, center_vertical=False,
                           colour=None, bg_colour=None, font_size=None, border_width=2, base=False):
        """
        Render word-wrapped text within a bounding rectangle.

        Args:
            text: String to render. Use '\\n' as a word token to force a line break.
            rect: pg.Rect defining the bounding area for the text.
            location: BlitLocation anchor point applied to rect.topleft.
            center_horizontal: If True, centre each line horizontally within rect.
            center_vertical: If True, centre the text block vertically within rect.
            colour: Colours enum value for the text. Defaults to Colours.hero_blue.
            bg_colour: Optional Colours enum value for a background fill behind the text block.
            font_size: Optional point size to use for rendering. Resets to normal after.
            border_width: Minimum padding in pixels between text and rect edges.
            base: If True, render onto the base layer.
        """
        rect: pg.Rect

        if colour is None:
            colour = Colours.hero_blue

        if font_size:
            self.fonts.update_custom(font_size)
            self.font = self.fonts.custom

        ids = [0]
        line_width = 0
        for idx, word in enumerate(text.split(" ")):
            if word == "\n":
                ids.append(idx)
                line_width = 0
            else:
                width = self.font.size(word + " ")[0]
                if line_width + self.font.size(word)[0] > rect.width - border_width*2:
                    ids.append(idx)
                    line_width = width
                else:
                    line_width += width
        ids.append(len(text.split(" ")))

        height, gap = 0, 10
        text_surfs = []
        for line in range(len(ids)-1):
            line_words = text.replace("\n ", "").split(" ")[ids[line]:ids[line+1]]
            line_text_surf = self.font.render(" ".join(line_words), True, colour.value)

            text_surfs.append(line_text_surf)

            height += line_text_surf.get_height() + gap  # cumulative height with 5px padding

        text_surf = pg.Surface(rect.size, pg.SRCALPHA)
        if bg_colour:
            text_surf.fill(bg_colour.value)
        total_height = sum([surf.get_height() for surf in text_surfs])
        total_height += gap*(len(text_surfs)-1)
        if center_vertical:
            y_offset = (rect.h - total_height) / 2
        else:
            y_offset = border_width

        for idx, surf in enumerate(text_surfs):
            if center_horizontal:
                text_surf.blit(surf, ((rect.width - surf.get_width())/2, y_offset + idx * (surf.get_height() + gap)))
            else:
                text_surf.blit(surf, (border_width, y_offset + idx*(surf.get_height() + gap)))

        blitPos = rect.topleft
        size = rect.size

        if location == BlitLocation.centre:
            blitPos -= size / 2
        elif location == BlitLocation.topRight:
            blitPos -= pg.Vector2(size.x, 0)
        elif location == BlitLocation.midTop:
            blitPos -= pg.Vector2(size.x / 2, 0)

        elif base:
            self.base_surface.blit(text_surf, blitPos)
        else:
            self.surface.blit(text_surf, blitPos)

        self.font = self.fonts.normal

    def create_layered_shape(self, pos, shape, size, number, colours, offsets,
                             radii, offsetWidth=False, offsetHeight=False, base=False):
        """
        Draw a multi-layer concentric shape (rectangle or ellipse) onto the screen.

        Args:
            pos: (x, y) position to blit the composed shape surface.
            shape: Shape type string — 'rectangle' or any other value for ellipse.
            size: Vector2 dimensions of the outer bounding surface.
            number: Number of concentric layers to draw.
            colours: List of Colours enum values, one per layer.
            offsets: List of Vector2 or Rect offsets applied cumulatively per layer.
            radii: List of border radii, one per layer (used for rectangles only).
            offsetWidth: Fraction of surface width to offset the blit position.
            offsetHeight: Fraction of surface height to offset the blit position.
            base: If True, blit onto the base layer.
        """
        # create surfaces
        surf = pg.Surface(size, pg.SRCALPHA)
        center = size / 2

        for layer in range(number):
            currentOffset = (0, 0)
            if type(offsets[0]) == pg.Vector2:
                currentOffset = pg.Vector2(0, 0)
            elif type(offsets[0]) == pg.Rect:
                currentOffset = pg.Rect(0, 0, 0, 0)

            for offset in offsets[0:layer + 1]:
                if type(offset) == pg.Vector2:
                    currentOffset += offset
                elif type(offset) == pg.Rect:
                    currentOffset = pg.Rect(pg.Vector2(currentOffset.topleft) + pg.Vector2(offset.topleft),
                                            pg.Vector2(currentOffset.size) + pg.Vector2(offset.size))

            rect = pg.Rect(0, 0, 0, 0)
            if type(currentOffset) == pg.Vector2:
                rect = pg.Rect((0, 0), size - 2 * currentOffset)
                rect.center = center
            elif type(currentOffset) == pg.Rect:
                rect = pg.Rect((currentOffset.x, currentOffset.w), (size.x - currentOffset.x - currentOffset.y,
                                                                    size.y - currentOffset.w - currentOffset.h))

            if shape == "rectangle":
                pg.draw.rect(surf, colours[layer], rect, border_radius=radii[layer])
            else:
                pg.draw.ellipse(surf, colours[layer].value, rect)

        offset = pg.Vector2(size)
        offset.x *= offsetWidth
        offset.y *= offsetHeight

        if base:
            self.base_surface.blit(surf, pos - offset)
        else:
            self.surface.blit(surf, pos - offset)

    def update_pixels(self, pos, colour=Colours.black.value, base=False, width=3):
        """
        Set individual pixels in a square region centred on pos.

        Args:
            pos: (x, y) centre position of the pixel patch.
            colour: RGBA tuple for the pixel colour.
            base: If True, write to the base layer.
            width: Side length in pixels of the square patch to fill.
        """
        pad = (width - 1) / 2
        for x_pos in range(int(pos[0] - pad), int(pos[0] + 1 + pad)):
            for y_pos in range(int(pos[1] - pad), int(pos[1] + 1 + pad)):
                if base:
                    self.base_surface.set_at((x_pos, y_pos), colour)
                else:
                    self.surface.set_at((x_pos, y_pos), colour)

    def add_speech_bubble(self, rect, pos, border=4, tiers=4, colour=Colours.black, base=False):
        """
        Draw a speech bubble border onto the screen.

        Args:
            rect: pg.Rect defining the bounding area of the bubble.
            pos: (x, y) position to blit the bubble surface.
            border: Total border thickness in pixels.
            tiers: Number of gradient tiers used to soften the border edges.
            colour: Colours enum value for the border.
            base: If True, blit onto the base layer.
        """
        rect: pg.Rect

        inside_tiers = max([2, math.floor(tiers/2)])
        blit_width = border / tiers
        surf = pg.Surface(rect.size, pg.SRCALPHA)

        tier_rect = rect
        tier_rect.topleft = (0, 0)
        # create the outside boundary
        for idx, tier in enumerate(range(tiers-1)):
            tier_width = (tiers - idx) * border / tiers

            top_line = pg.Rect((tier_width, tier_rect.top),
                               (tier_rect.width - (1-idx/(tiers-idx))*tier_width * 2, tier_width))
            bottom_line = pg.Rect((tier_width, tier_rect.top + tier_rect.height - tier_width),
                                  (tier_rect.width - (1-idx/(tiers-idx))*tier_width * 2, tier_width))
            left_line = pg.Rect((tier_rect.left, tier_width), (tier_width, tier_rect.height - (1-idx/(tiers-idx))* 2 * tier_width))
            right_line = pg.Rect((tier_rect.left + tier_rect.width - tier_width, tier_width),
                                 (tier_width, tier_rect.height - 2 * (1-idx/(tiers-idx))*tier_width))
            border_lines = [top_line, bottom_line, left_line, right_line]

            for line in border_lines:
                pg.draw.rect(surf, colour.value, line)

            tier_rect = tier_rect.inflate(-(2 * border) / tiers, - (2 * border) / tiers)

        # create inside border
        tier_rect = rect
        tier_rect.topleft = (0, 0)
        tier_rect = tier_rect.inflate(-border*2, -border*2)

        for idx, count in enumerate(range(inside_tiers)):
            top_line_1 = pg.Rect(tier_rect.topleft,
                                 ((inside_tiers-idx)*blit_width, blit_width))
            top_line_2 = pg.Rect(tier_rect.topright - pg.Vector2((inside_tiers - idx) * blit_width, 0),
                                 ((inside_tiers - idx) * blit_width, blit_width))

            bottom_line_1 = pg.Rect(tier_rect.bottomleft - pg.Vector2(0, blit_width),
                                    ((inside_tiers-idx)*blit_width, blit_width))
            bottom_line_2 = pg.Rect(tier_rect.bottomright - pg.Vector2((inside_tiers - idx) * blit_width, blit_width),
                                 ((inside_tiers - idx) * blit_width, blit_width))

            tier_rect = tier_rect.inflate(0, -blit_width * 2)

            pg.draw.rect(surf, colour.value, top_line_1)
            pg.draw.rect(surf, colour.value, top_line_2)
            pg.draw.rect(surf, colour.value, bottom_line_1)
            pg.draw.rect(surf, colour.value, bottom_line_2)

        if base:
            self.base_surface.blit(surf, pos)
        else:
            self.surface.blit(surf, pos)

    def refresh(self):
        """Clear the dynamic and sprite layers, preserving the base layer."""
        self.surface = pg.Surface(self.size, pg.SRCALPHA)
        self.sprite_surface = pg.Surface(self.size, pg.SRCALPHA)

    def clear_surfaces(self):
        """Release all surface references and the active font."""
        self.surface = None
        self.base_surface = None
        self.font = None

    def get_surface(self):
        """
        Composite all three layers into a single surface.

        Returns:
            pg.Surface with base, dynamic, and sprite layers merged top-to-bottom.
        """
        display_surf = self.base_surface.copy()
        display_surf.blit(self.surface, (0, 0))
        display_surf.blit(self.sprite_surface, (0, 0))

        return display_surf

    def scale_surface(self, scale, base=False):
        """
        Scale the dynamic (and optionally base) surface by a scalar factor.

        Args:
            scale: Scalar multiplier applied to current surface dimensions.
            base: If True, also scale the base surface.
        """
        self.size = pg.Vector2(self.base_surface.get_size()) * scale
        self.surface = pg.transform.scale(self.surface, pg.Vector2(self.surface.get_size()) * scale)
        if base:
            self.base_surface = pg.transform.scale(self.base_surface, pg.Vector2(self.base_surface.get_size()) * scale)
