import os
import random
import re
import time

import pygame as pg

from consultation.screen import Screen, Colours


class Avatar:
    def __init__(self, size=pg.Vector2(420, 420), skin_tone=None):
        avatar_base_path = "consultation/resources/graphics/avatars"
        avatar_paths = [os.path.join(avatar_base_path, avatar_file) for avatar_file in
                        os.listdir(avatar_base_path) if (".png" in avatar_file)]

        # gender =
        # self.image = pg.image.load(f"hero/consultation/resources/graphics/sprites/avatar_{gender}.png")

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

        self.mouth_sprites = [pg.image.load(f"consultation/resources/graphics/sprites/mouths/mouth_{idx}.png")
                              for idx in range(1, 13)]

        if scale.x != 1 or scale.y != 1:
            self.mouth_sprites = [pg.transform.scale_by(surface, scale) for surface in self.mouth_sprites]

        self.mouth_idx = 0

    def get_surface(self):
        surface = self.image.copy()
        if self.mouth_idx > 0:
            surface.blit(self.mouth_sprites[self.mouth_idx], (0, 0))
        return surface
