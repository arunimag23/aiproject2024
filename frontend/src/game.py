import pygame 
import random

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 700
GRID_SIZE = 4
WORD_BOX_SIZE = 120
PADDING = 10
MESSAGE_HEIGHT = 50  

# Colors
WHITE = (255, 255, 255)
RED = (224, 76, 76)
YELLOW = (255, 255, 0)
BLUE = (137, 207, 240)
GREEN = (111,148,96)
BLACK = (0, 0, 0)
BORDER_COLOR = (128, 128, 128) 

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Connections Game")

# Font
font = pygame.font.Font(None, 36)

# Game data
words = [
    'Strawberry', 'Kiwi', 'Spring', 'Summer',
    'Tennis', 'Autumn', 'Winter', 'Soccer',
    'Indian', 'Thai', 'Basketball', 'Japanese',
    'Volleyball', 'Apple', 'Chinese', 'Banana'
]
random.shuffle(words)

categories = {
    'Seasons': RED,
    'Sports': YELLOW,
    'Cuisines': BLUE,
    'Fruits': GREEN
}

correct_category = {
    'Spring': 'Seasons', 'Summer': 'Seasons', 'Autumn': 'Seasons', 'Winter': 'Seasons',
    'Soccer': 'Sports', 'Basketball': 'Sports', 'Tennis': 'Sports', 'Volleyball': 'Sports',
    'Indian': 'Cuisines', 'Thai': 'Cuisines', 'Chinese': 'Cuisines', 'Japanese': 'Cuisines',
    'Strawberry': 'Fruits', 'Kiwi': 'Fruits', 'Apple': 'Fruits', 'Banana': 'Fruits'
}

# Game state
selected_word = None
word_positions = {}
message = "Click a word to begin."

# Helper functions
def draw_word_box(word, x, y, color=WHITE):
    pygame.draw.rect(screen, color, (x, y, WORD_BOX_SIZE, WORD_BOX_SIZE))
    pygame.draw.rect(screen, BORDER_COLOR, (x, y, WORD_BOX_SIZE + 20, WORD_BOX_SIZE), 2)
    text = font.render(word, True, BLACK)
    text_rect = text.get_rect(center=(x + (WORD_BOX_SIZE + 20) // 2, y + WORD_BOX_SIZE // 2))
    screen.blit(text, text_rect)

def draw_category_box(category, x, y, color):
    pygame.draw.rect(screen, color, (x, y, WIDTH // 4 - PADDING, 50))
    text = font.render(category, True, BLACK)
    text_rect = text.get_rect(center=(x + (WIDTH // 4 - PADDING) // 2, y + 25))
    screen.blit(text, text_rect)

def display_message(text):
    pygame.draw.rect(screen, WHITE, (0, HEIGHT - MESSAGE_HEIGHT, WIDTH, MESSAGE_HEIGHT))
    msg_text = font.render(text, True, BLACK)
    text_rect = msg_text.get_rect(center=(WIDTH // 2, HEIGHT - MESSAGE_HEIGHT // 2))
    screen.blit(msg_text, text_rect)  

def check_word_click(pos):
    global selected_word, message
    for word, (x, y, _) in word_positions.items():  
        if x < pos[0] < x + WORD_BOX_SIZE and y < pos[1] < y + WORD_BOX_SIZE:
            selected_word = word
            return True
    return False

def check_category_click(pos):
    global selected_word, message
    if selected_word:
        x, y = pos
        for index, category in enumerate(categories):
            category_x = index * (WIDTH // 4)
            category_y = HEIGHT - 60 - MESSAGE_HEIGHT - PADDING
            if category_x < x < category_x + (WIDTH // 4) and category_y < y < category_y + 50:
                if correct_category[selected_word] == category:
                    word_positions[selected_word] = (word_positions[selected_word][0], 
                                                     word_positions[selected_word][1], 
                                                     categories[category])
                    message = f"Correct! {selected_word} is in {category}."
                else:
                    message = f"Incorrect! {selected_word} does not belong to {category}."
                selected_word = None
                return True
    return False

# Main game 
running = True
try:
    while running:
        screen.fill(WHITE)
    
        # Draw the grid of words
        for i, word in enumerate(words):
            row, col = divmod(i, GRID_SIZE)
            x = col * (WORD_BOX_SIZE + PADDING) + PADDING
            y = row * (WORD_BOX_SIZE + PADDING) + PADDING
            color = word_positions.get(word, (0, 0, WHITE))[2]
            draw_word_box(word, x, y, color)
            word_positions[word] = (x, y, color)
    
        # Draw categories
        category_y = HEIGHT - 60 - MESSAGE_HEIGHT - PADDING  # Adjusted position
        for i, (category, color) in enumerate(categories.items()):
            draw_category_box(category, i * (WIDTH // 4), category_y, color)

        # Display message
        display_message(message)
    
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # Check for word click
                if not check_word_click(pos):
                    check_category_click(pos)
    
        pygame.display.flip()

except Exception as e:
    print(f"An error occurred: {e}")

pygame.quit()