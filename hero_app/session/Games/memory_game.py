import pygame
import random
import sys
import os

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 750  # Increased to accommodate larger icon preview area
GRID_SIZE = 3
CELL_SIZE = 120
GRID_PADDING = 10
ENCODING_TIME = 8000  # 8 seconds in milliseconds
ICON_SIZE = 90  # Size to scale icons to

# Colors
WHITE = (255, 255, 255)
LIGHT_GRAY = (220, 220, 220)
DARK_GRAY = (158, 158, 158)
GREEN = (76, 175, 80)
RED = (244, 67, 54)
BLACK = (0, 0, 0)
TEXT_GRAY = (100, 100, 100)

def load_icons():
    """Load all icon images from the Icons folder"""
    icons = []
    icons_folder = "Memory Game/Icons"

    if not os.path.exists(icons_folder):
        print(f"ERROR: '{icons_folder}' folder not found!")
        sys.exit(1)

    # Get all image files from Icons folder
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    icon_files = [f for f in os.listdir(icons_folder) 
                  if f.lower().endswith(image_extensions)]

    if len(icon_files) < 9:
        print(f"ERROR: Need at least 9 images in '{icons_folder}' folder. Found {len(icon_files)}.")
        sys.exit(1)

    # Load and scale first 9 images
    for i, filename in enumerate(sorted(icon_files)[:9]):
        filepath = os.path.join(icons_folder, filename)
        try:
            image = pygame.image.load(filepath)
            # Scale to fit in cell with padding
            scaled_image = pygame.transform.smoothscale(image, (ICON_SIZE, ICON_SIZE))
            icons.append(scaled_image)
            print(f"Loaded: {filename}")
        except Exception as e:
            print(f"ERROR loading {filename}: {e}")
            sys.exit(1)

    print(f"Successfully loaded {len(icons)} icons")
    return icons

class MemoryGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Memory Game - Medical Device")
        self.clock = pygame.time.Clock()

        # Load icons
        self.icons = load_icons()

        # Fonts
        self.title_font = pygame.font.SysFont('Arial', 28, bold=True)
        self.instruction_font = pygame.font.SysFont('Arial', 18)
        self.score_font = pygame.font.SysFont('Arial', 16)

        # Game state
        self.phase = "waiting"  # waiting, encoding, recall, feedback
        self.symbol_positions = {}
        self.recalled_symbols = set()
        self.incorrect_symbols = set()  # Track which symbols were answered incorrectly
        self.current_symbol_index = 0
        self.symbols_to_recall = []
        self.score = 0
        self.total_trials = 0
        self.encoding_start_time = 0
        self.feedback_start_time = 0
        self.correct_pos = None
        self.wrong_pos = None
        self.button_rect = None

        # Grid position calculation - moved down to make room for larger icon preview
        grid_total_width = GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * GRID_PADDING
        self.grid_offset_x = (WINDOW_WIDTH - grid_total_width) // 2
        self.grid_offset_y = 220  # Increased from 150 to make room for icon preview

    def start_game(self):
        """Initialize a new game round"""
        self.phase = "encoding"
        self.symbol_positions = {}
        self.recalled_symbols = set()
        self.incorrect_symbols = set()
        self.current_symbol_index = 0
        self.score = 0
        self.total_trials = 0
        self.correct_pos = None
        self.wrong_pos = None

        # Randomize icon positions
        positions = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
        random.shuffle(positions)

        for idx in range(len(self.icons)):
            self.symbol_positions[idx] = positions[idx]

        # Create randomized recall order
        self.symbols_to_recall = list(range(len(self.icons)))
        random.shuffle(self.symbols_to_recall)

        self.encoding_start_time = pygame.time.get_ticks()

    def get_cell_rect(self, row, col):
        """Get the rectangle for a grid cell"""
        x = self.grid_offset_x + col * (CELL_SIZE + GRID_PADDING)
        y = self.grid_offset_y + row * (CELL_SIZE + GRID_PADDING)
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    def get_clicked_cell(self, pos):
        """Convert mouse position to grid cell, or None if outside grid"""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.get_cell_rect(row, col).collidepoint(pos):
                    return (row, col)
        return None

    def draw_grid(self):
        """Draw the game grid"""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                rect = self.get_cell_rect(row, col)

                # Determine cell color and content
                color = DARK_GRAY
                icon_to_draw = None

                if self.phase == "encoding":
                    # Show all icons during encoding
                    for icon_idx, (s_row, s_col) in self.symbol_positions.items():
                        if (s_row, s_col) == (row, col):
                            icon_to_draw = self.icons[icon_idx]
                            color = (117, 117, 117)
                            break

                elif self.phase == "recall" or self.phase == "feedback":
                    # Show recalled icons with appropriate colors
                    for icon_idx in self.recalled_symbols:
                        s_row, s_col = self.symbol_positions[icon_idx]
                        if (s_row, s_col) == (row, col):
                            icon_to_draw = self.icons[icon_idx]
                            # Red if incorrect, green if correct
                            color = RED if icon_idx in self.incorrect_symbols else GREEN
                            break

                    # Show feedback
                    if self.phase == "feedback":
                        if self.correct_pos and (row, col) == self.correct_pos:
                            # Show correct location - red if wrong answer, will be green next phase if correct
                            for icon_idx, (s_row, s_col) in self.symbol_positions.items():
                                if (s_row, s_col) == (row, col):
                                    icon_to_draw = self.icons[icon_idx]
                                    color = RED if icon_idx in self.incorrect_symbols else GREEN
                                    break
                        elif self.wrong_pos and (row, col) == self.wrong_pos:
                            color = RED

                # Draw cell
                pygame.draw.rect(self.screen, color, rect, border_radius=8)
                pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=8)

                # Draw icon if needed
                if icon_to_draw:
                    icon_rect = icon_to_draw.get_rect(center=rect.center)
                    self.screen.blit(icon_to_draw, icon_rect)

    def draw_ui(self):
        """Draw UI elements (title, instructions, score)"""
        # Title
        if self.phase == "waiting":
            title_text = "Memory Game"
        elif self.phase == "encoding":
            title_text = "Encoding phase"
        else:
            title_text = "Recall phase"

        title = self.title_font.render(title_text, True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 40))
        self.screen.blit(title, title_rect)

        # Instructions with icon preview
        if self.phase == "waiting":
            instruction = "Click START GAME to begin"
            instr = self.instruction_font.render(instruction, True, TEXT_GRAY)
            instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, 90))
            self.screen.blit(instr, instr_rect)

        elif self.phase == "encoding":
            elapsed = (pygame.time.get_ticks() - self.encoding_start_time) / 1000
            remaining = max(0, 8 - int(elapsed))
            instruction = f"Remember the symbol locations ({remaining}s)"
            instr = self.instruction_font.render(instruction, True, TEXT_GRAY)
            instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, 90))
            self.screen.blit(instr, instr_rect)

        elif self.phase == "recall" or self.phase == "feedback":
            if self.current_symbol_index < len(self.symbols_to_recall):
                instruction = "Where was this symbol?"
                instr = self.instruction_font.render(instruction, True, TEXT_GRAY)
                instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, 90))
                self.screen.blit(instr, instr_rect)

                # Show current icon to recall - larger area (moved down and more space)
                current_icon = self.icons[self.symbols_to_recall[self.current_symbol_index]]
                icon_rect = current_icon.get_rect(center=(WINDOW_WIDTH // 2, 150))
                self.screen.blit(current_icon, icon_rect)
            else:
                instruction = "All symbols recalled!"
                instr = self.instruction_font.render(instruction, True, TEXT_GRAY)
                instr_rect = instr.get_rect(center=(WINDOW_WIDTH // 2, 90))
                self.screen.blit(instr, instr_rect)

        # Score
        if self.total_trials > 0:
            percentage = int((self.score / self.total_trials) * 100)
            score_text = f"Score: {self.score}/{self.total_trials} ({percentage}%)"
        else:
            score_text = "Score: 0/0"

        score = self.score_font.render(score_text, True, TEXT_GRAY)
        score_rect = score.get_rect(center=(WINDOW_WIDTH // 2, 690))
        self.screen.blit(score, score_rect)

        # Start/Restart button
        self.button_rect = None
        if self.phase == "waiting" or (self.phase == "recall" and 
                                       self.current_symbol_index >= len(self.symbols_to_recall)):
            self.button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 100, 630, 200, 50)
            pygame.draw.rect(self.screen, GREEN, self.button_rect, border_radius=10)

            button_text_str = "START GAME" if self.phase == "waiting" else "PLAY AGAIN"
            button_text = self.instruction_font.render(button_text_str, True, WHITE)
            button_text_rect = button_text.get_rect(center=self.button_rect.center)
            self.screen.blit(button_text, button_text_rect)

    def handle_click(self, pos):
        """Handle mouse clicks"""
        # Start button click
        if self.button_rect and self.button_rect.collidepoint(pos):
            self.start_game()
            return

        # Grid click during recall phase
        if self.phase == "recall":
            cell = self.get_clicked_cell(pos)
            if cell is not None:
                self.handle_recall_click(cell)

    def handle_recall_click(self, clicked_cell):
        """Handle a recall attempt"""
        if self.current_symbol_index >= len(self.symbols_to_recall):
            return

        symbol_idx = self.symbols_to_recall[self.current_symbol_index]
        correct_cell = self.symbol_positions[symbol_idx]

        self.total_trials += 1

        if clicked_cell == correct_cell:
            # Correct answer
            self.score += 1
            self.recalled_symbols.add(symbol_idx)
            self.correct_pos = correct_cell
            self.wrong_pos = None
        else:
            # Wrong answer - mark this symbol as incorrect
            self.incorrect_symbols.add(symbol_idx)
            self.recalled_symbols.add(symbol_idx)
            self.correct_pos = correct_cell
            self.wrong_pos = clicked_cell

        self.phase = "feedback"
        self.feedback_start_time = pygame.time.get_ticks()

    def update(self):
        """Update game state"""
        if self.phase == "encoding":
            # Check if encoding time is over
            if pygame.time.get_ticks() - self.encoding_start_time >= ENCODING_TIME:
                self.phase = "recall"

        elif self.phase == "feedback":
            # Show feedback for 1.5 seconds, then move to next symbol
            if pygame.time.get_ticks() - self.feedback_start_time >= 1500:
                self.current_symbol_index += 1
                self.phase = "recall"
                self.correct_pos = None
                self.wrong_pos = None

    def run(self):
        """Main game loop"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)

            # Update
            self.update()

            # Draw
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

# Run the game
if __name__ == "__main__":
    game = MemoryGame()
    game.run()