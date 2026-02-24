import os
import sys
import pygame as pg

os.environ['SDL_VIDEO_WINDOW_POS'] = "1029,600"  # Top physical screen (HDMI-A-2)

pg.init()
pg.font.init()
pg.event.pump()

# Add hero_system root to path so db/ module is found
sys.path.insert(0, '/home/hero/HERO-Device/hero_system')

from hero.consultation.orchestrator import Consultation
from db.db_access import HeroDB

# Connect to DB
try:
    db = HeroDB()
    print("✓ Connected to hero_db")
except Exception as e:
    print(f"⚠ Could not connect to DB, running without: {e}")
    db = None

consult = Consultation(
    pi=False,         # Change to True on the Pi
    authenticate=True,
    seamless=True,
    local=True,       # Also save locally as backup
    enable_speech=False,
    scale=0.8,
    db=db,
)

consult.loop()

if db:
    db.close()
    
