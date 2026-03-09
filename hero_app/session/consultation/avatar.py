"""
Avatar sprite for the HERO consultation display.

Loads a randomised avatar image and mouth animation sprites used by
the DisplayScreen during text-to-speech playback.
"""

import os
import random

import pygame as pg

from consultation.screen import Screen, Colours


class Avatar:
    """
    Animated avatar composed of a base image and swappable mouth sprites.

    A random avatar image is selected from the avatars resource folder on
    initialisation. Mouth sprites are cycled during speech via mouth_idx.

    Attributes:
        image: Scaled base avatar surface.
        size: Vector2 dimensions of the avatar.
        mouth_sprites: List of 12 mouth position surfaces.
        mouth_idx: Index of the currently active mouth sprite (0 = closed).
    """

    def __init__(self, size=pg.Vector2(420, 420), skin_tone=None):
        """
        Initialise the avatar by loading a random base image and mouth sprites.

        Args:
            size: Vector2 or tuple (width, height) to scale the avatar to.
            skin_tone: Unused; reserved for future skin tone filtering.
        """
        avatar_base_path = "hero/consultation/resources/graphics/avatars"
        avatar_paths = [os.path.join(avatar_base_path, avatar_file) for avatar_file in
                        os.listdir(avatar_base_path) if (".png" in avatar_file)]

        if skin_tone:
            self.image = pg.image.load(random.choice(avatar_paths))
        else:
            self.image = pg.image.load(random.choice(avatar_paths))

        if size:
            self.size = pg.Vector2(size)
            prev_size = pg.Vector2(self.image.get_size())
            self.image = pg.transform.scale(self.image, size)
            scale = pg.Vector2(self.size.x / prev_size.x, self.size.y / prev_size.y)
        else:
            self.size = pg.Vector2(self.image.get_size())
            scale = pg.Vector2(1, 1)

        self.mouth_sprites = [pg.image.load(f"hero/consultation/resources/graphics/sprites/mouths/mouth_{idx}.png")
                              for idx in range(1, 13)]

        if scale.x != 1 or scale.y != 1:
            self.mouth_sprites = [pg.transform.scale_by(surface, scale) for surface in self.mouth_sprites]

        self.mouth_idx = 0

    def get_surface(self):
        """
        Compose the avatar surface with the current mouth sprite overlaid.

        Returns:
            pg.Surface with the base avatar and active mouth sprite blitted on top.
        """
        surface = self.image.copy()
        if self.mouth_idx > 0:
            surface.blit(self.mouth_sprites[self.mouth_idx], (0, 0))
        return surface
