import os
import pygame as pg

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

pg.init()
pg.font.init()
pg.event.pump()

from hero.consultation.orchestrator import Consultation

consult = Consultation(
    pi=False,         # Change to True on the Pi
    authenticate=True,
    seamless=True,
    local=True,
    enable_speech=False,
    scale=0.7
)
consult.loop()
