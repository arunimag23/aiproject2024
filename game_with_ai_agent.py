import pygame
import random
import numpy as np
from aigame import viterbi_with_constraints  # Import your Viterbi algorithm
from aigame import emission_matrix_observations_states, observations, states, transition_matrix

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
GREEN = (111, 148, 96)
BLACK = (0, 0, 0)
BORDER_COLOR = (128, 128, 128)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Connections Game")

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

# Game variables
word_positions = {}
message = "AI is classifying..."
ai_progress = 0  # Counter to track AI progress
best_path = []  # This will store the step-by-step classifications


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

def ai_play_step():
    """Process one step of AI classification using Viterbi."""
    global ai_progress, message, best_path
    if ai_progress < len(words):
        # Process the next word
        word = words[ai_progress]
        category = best_path[ai_progress]
        color = categories[category]

        # Update the word's box with the category color
        x, y, _ = word_positions[word]
        word_positions[word] = (x, y, color)

        # Check correctness and update the message
        correct = correct_category[word] == category
        if correct:
            message = f"Correct! '{word}' classified as {category}."
        else:
            message = f"Incorrect! '{word}' classified as {category} instead of {correct_category[word]}."

        # Move to the next word
        ai_progress += 1
    else:
        message = "AI has finished classifying all words!"

def draw_game_state():
    """Redraw the entire game state."""
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
    pygame.display.flip()

# Initialize the game state
for i, word in enumerate(words):
    row, col = divmod(i, GRID_SIZE)
    x = col * (WORD_BOX_SIZE + PADDING) + PADDING
    y = row * (WORD_BOX_SIZE + PADDING) + PADDING
    word_positions[word] = (x, y, WHITE)

# Run the Viterbi algorithm
best_path = viterbi_with_constraints(observations, states, emission_matrix_observations_states, transition_matrix)

# Main game loop
running = True

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Process the AI's next step
        ai_play_step()

        # Redraw the game state
        draw_game_state()
        pygame.time.delay(500)  # Add delay for visualization

except Exception as e:
    print(f"An error occurred: {e}")

pygame.quit()