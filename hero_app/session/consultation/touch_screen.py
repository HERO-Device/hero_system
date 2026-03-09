"""
Bottom touchscreen surface and interactive sprite primitives.

Provides GameButton, GameObjects, and TouchScreen — the interactive layer
rendered on the lower physical screen during patient consultations and games.
"""

import pygame as pg

from consultation.screen import Screen, BlitLocation, Colours


class GameButton(pg.sprite.Sprite):
    """
    A clickable rectangular button sprite.

    Attributes:
        object_type: Always 'button', used by GameObjects.draw() for dispatch.
        rect: pg.Rect defining the button's position and size.
        id: Identifier returned by click_return() when clicked.
        colour: RGBA tuple used to fill the button background.
        text: Optional label rendered centred on the button.
        label: Optional secondary label rendered above the button.
    """

    def __init__(self, position, size, id, text=None, label=None, colour=None):
        """
        Initialise a button at the given position.

        Args:
            position: (x, y) top-left position of the button.
            size: (width, height) dimensions of the button.
            id: Value returned when the button is clicked.
            text: Optional text rendered centred on the button face.
            label: Optional secondary text rendered above the button.
            colour: Optional Colours enum value for the button fill. Defaults to hero_blue.
        """
        super().__init__()
        self.object_type = "button"
        self.rect = pg.Rect(position, size)
        self.id = id
        if colour:
            self.colour = colour.value
        else:
            self.colour = Colours.hero_blue.value
        self.text = text
        self.label = label

    def is_clicked(self, pos):
        """
        Test whether a screen position falls within the button rect.

        Args:
            pos: (x, y) position to test.

        Returns:
            True if pos is within the button rect, False otherwise.
        """
        if self.rect.collidepoint(pos):
            return True
        else:
            return False

    def click_return(self):
        """
        Return the button's identifier on click.

        Returns:
            The id value assigned at initialisation.
        """
        return self.id


class GameObjects(pg.sprite.Group):
    """
    Sprite group that renders HERO game objects onto a Screen layer.

    Handles buttons, cards, circles, and clock hands via type dispatch
    in the draw() method.
    """

    def __init__(self, sprites):
        """
        Initialise the group with an initial list of sprites.

        Args:
            sprites: List of sprite objects to add to the group.
        """
        super().__init__(self, sprites)

    def draw(self, screen: Screen, bgsurf=None, special_flags: int = 0):
        """
        Render all sprites in the group onto the screen's sprite layer.

        Args:
            screen: Screen instance to render onto.
            bgsurf: Unused; retained for compatibility with pg.sprite.Group.draw().
            special_flags: Unused; retained for compatibility with pg.sprite.Group.draw().
        """
        for obj in self.sprites():
            if obj.object_type == "button":
                pg.draw.rect(screen.sprite_surface, obj.colour, obj.rect, border_radius=16)
                if obj.text:
                    screen.add_text(obj.text, colour=Colours.white, location=BlitLocation.centre, pos=obj.rect.center,
                                    sprite=True)
                if obj.label:
                    screen.add_text(obj.label, colour=Colours.darkGrey, location=BlitLocation.midBottom, pos=obj.rect.midtop,
                                    sprite=True)

            elif obj.object_type == "card" or obj.object_type == "circle":
                screen.add_surf(obj.image, pos=obj.rect.topleft, sprite=True)

            elif obj.object_type == "clock_hand":
                screen.add_surf(obj.image, screen.size / 2, location=BlitLocation.centre, sprite=True)


class TouchScreen(Screen):
    """
    Lower consultation screen handling touch/click input and sprite rendering.

    Extends Screen with a sprite group and click-test routing. Supports a
    power-off mode that replaces the display with a splash screen.

    Attributes:
        sprites: GameObjects group of currently active interactive sprites.
        power_off: If True, get_surface() returns the power-off splash screen.
    """

    def __init__(self, size, colour=Colours.white):
        """
        Initialise the touch screen and build the power-off splash surface.

        Args:
            size: Tuple or Vector2 (width, height) of the screen.
            colour: Background Colours enum value. Defaults to white.
        """
        super().__init__(size, colour=colour)
        self.sprites = GameObjects([])
        self.power_off = False

        self.power_off_surface = pg.Surface((self.size.x, self.size.y), pg.SRCALPHA)
        self.power_off_surface.fill(Colours.white.value)

        image = pg.image.load("hero/consultation/resources/graphics/logo.png")
        image = pg.transform.scale(image, size=self.size.yy * 0.8)
        pos = self.size / 2

        imageRect = pg.Rect(pos, image.get_size())
        imageRect.topleft = pg.Vector2(imageRect.topleft) - pg.Vector2(imageRect.size) / 2

        self.power_off_surface.blit(image, imageRect)

    def click_test(self, pos):
        """
        Test whether pos hits any active sprite and return its id.

        Args:
            pos: (x, y) position in bottom-screen coordinates.

        Returns:
            The clicked sprite's click_return() value, or None if no hit.
        """
        if self.sprites:
            for sprite in self.sprites:
                if sprite.is_clicked(pos):
                    return sprite.click_return()

        return None

    def kill_sprites(self):
        """Remove all sprites from the active group."""
        self.sprites.empty()

    def get_surface(self):
        """
        Composite the touch screen into a single surface for blitting.

        Returns:
            pg.Surface with all sprites rendered, or the power-off splash if power_off is True.
        """
        if self.power_off:
            return self.power_off_surface

        self.sprites.draw(self)
        display_surf = self.base_surface.copy()
        display_surf.blit(self.surface, (0, 0))
        display_surf.blit(self.sprite_surface, (0, 0))

        return display_surf

    def get_object(self, object_id):
        """
        Retrieve a sprite from the group by its id.

        Args:
            object_id: The id value to search for.

        Returns:
            The matching sprite, or None if not found.
        """
        for game_object in self.sprites:
            if game_object.id == object_id:
                return game_object
