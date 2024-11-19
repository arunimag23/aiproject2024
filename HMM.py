import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api

word2vec_model = api.load("word2vec-google-news-300")

def get_word_embedding(word):
    try:
        return word2vec_model[word]
    except KeyError:
        print(f"The word '{word}' is not in the Word2Vec vocabulary.")
        return np.zeros(word2vec_model.vector_size)

# Calculate cosine similarity
def word2vec_calculation(word1, word2):
    embedding1 = get_word_embedding(word1).reshape(1, -1)
    embedding2 = get_word_embedding(word2).reshape(1, -1)
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    if similarity < 0:
        return similarity**2 # Ensuring similarity is non-negative
    else:
        return similarity 


# Preprocess state names by removing numeric suffixes
def preprocess_states(states):
    cleaned_states = [re.sub(r'\d+', '', state) for state in states]
    return cleaned_states

# Calculate the probability of a word belonging to a state
def prob_in_state(observation, theta_of_similarity):
    thetas_for_categories = []
    for index in range(len(categories)):
        co_similarity = word2vec_calculation(observation, categories[index])
        thetas_for_categories.append(co_similarity)
    probability = theta_of_similarity / np.sum(thetas_for_categories)
    return probability

# Define observations, categories, and states
observations = np.array(["Strawberry", "Kiwi", "Spring", "Summer", 
                         "Tennis", "Fall", "Winter", "Soccer",
                         "Indian", "Thai", "Basketball", "Japanese",
                         "Volleyball", "Apple", "Chinese", "Banana"])

categories = np.array(["Fruit", "Season", "Sport", "Cuisine"])

states = np.array(["Fruit1", "Fruit2", "Fruit3", "Fruit4", 
                   "Season1", "Season2", "Season3", "Season4",
                   "Sport1", "Sport2", "Sport3", "Sport4",
                   "Cuisine1", "Cuisine2", "Cuisine3", "Cuisine4"])

# Define category-state mapping
category_state_mapping = {
    "Fruits": states[:4],
    "Seasons": states[4:8],
    "Sports": states[8:12],
    "Cuisines": states[12:]
}

def hash_game_state(game_state):
    hashed_state = int("".join(map(str, game_state)))
    return hashed_state

def dehash_game_state(hashed_state):
    game_state = [int(digit) for digit in str(hashed_state)]
    return game_state

# Create the emission matrix
emission_matrix_observations_states = np.zeros((len(observations), len(states)))
cleaned_states = preprocess_states(states)

for observation_index in range(len(observations)):
    for state_index in range(len(cleaned_states)):
        similarity = word2vec_calculation(observations[observation_index], cleaned_states[state_index])
        probability = prob_in_state(observations[observation_index], similarity)
        emission_matrix_observations_states[observation_index][state_index] = probability / 4  

#print(emission_matrix_observations_states)

#DUMMY TRANSITION MATRIX DOESN"T WORK
# Transition matrix - favor transitions within the same category
transition_matrix = np.zeros((len(states), len(states)))
for i, current_state in enumerate(states):
    for j, next_state in enumerate(states):
        # Transitions are more probable within the same category (e.g., Fruit1 -> Fruit2)
        if current_state[:-1] == next_state[:-1]:  # Same category
            transition_matrix[i, j] = 0.8  # Higher probability for within-category transitions
        else:
            transition_matrix[i, j] = 0.2 / (len(states) - 4)  # Lower probability for across-category transitions

# Normalize transition matrix
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

# IMPLEMENT VIRTERBI ALGORITHM HERE:

# Viterbi algorithm
def viterbi_with_constraints(observations, states, emission_matrix, transition_matrix):
    n_observations = len(observations)
    n_states = len(states)

    # Initialize DP table and backpointer
    dp = np.zeros((n_states, n_observations))
    backpointer = np.zeros((n_states, n_observations), dtype=int)

    # Track which states are available
    available_states = [True] * n_states

    # Initialize base cases for the first observation
    for s in range(n_states):
        dp[s, 0] = emission_matrix[0, s] if available_states[s] else 0
        backpointer[s, 0] = -1

    # Viterbi with constraints
    for t in range(1, n_observations):
        for s in range(n_states):
            if not available_states[s]:
                dp[s, t] = 0  #if state is already filled, can't use it again.
                continue

            # Calculate probabilities based on previous states
            best_prob = -1
            best_prev_state = -1
            for prev_s in range(n_states):
                if available_states[prev_s] or t == 1:  # Ensure only unfilled states are considered
                    prob = dp[prev_s, t-1] * transition_matrix[prev_s, s] * emission_matrix[t, s]
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_state = prev_s 

            dp[s, t] = best_prob
            backpointer[s, t] = best_prev_state

        # Mark the best state for the current observation as filled
        best_state = np.argmax(dp[:, t])
        available_states[best_state] = False

    # Backtrace to find the best path
    best_path = np.zeros(n_observations, dtype=int)
    best_path[-1] = np.argmax(dp[:, -1])
    
    for t in range(n_observations - 2, -1, -1):
        best_path[t] = backpointer[best_path[t + 1], t + 1]

    # Convert state indices to state names
    best_state_sequence = [states[state_idx] for state_idx in best_path]
    return best_state_sequence

# Run the Viterbi algorithm with constraints
best_path = viterbi_with_constraints(observations, states, emission_matrix_observations_states, transition_matrix)

# Output results
print("Best path of states for given observations:")
for obs, state in zip(observations, best_path):
    print(f"{obs} -> {state}")

# # #TO DO: create transition matrix
# transition_matrix = np.zeros([4444, 4444], dtype=int)

# for row in range(len(transition_matrix)):
#     for column in range(len(transition_matrix)):
#             current_state = dehash_game_state(column)
#             for i in range(len(current_state)):
#                 current_state[i] += 1
#                 hashed_state = hash_game_state(current_state)
#                 if hashed_state < 4445:
#                     transition_matrix[row][hashed_state] = 1
#                     current_state[i] -= 1
#                 else:
#                     transition_matrix[row][hashed_state] = 0

# print("Emission Matrix (Categories x States):")
# print(emission_matrix_observations_states)

# Example similarity calculations
similarity_fruit = word2vec_calculation('Strawberry', 'Fruit')
similarity_season = word2vec_calculation('Strawberry', 'Season')
similarity_sport = word2vec_calculation('Strawberry', 'Sport')
similarity_cuisine = word2vec_calculation('Strawberry', 'Cuisine')

similarity_fruit_kiwi = word2vec_calculation('Kiwi', 'Fruit')
similarity_season_kiwi = word2vec_calculation('Kiwi', 'Season')
similarity_sport_kiwi = word2vec_calculation('Kiwi', 'Sport')
similarity_cuisine_kiwi = word2vec_calculation('Kiwi', 'Cuisine')

similarity_fruit_strawberry = word2vec_calculation('Strawberry', 'Strawberry')

print(f"Cosine similarity between 'Strawberry' and 'Strawberry': {similarity_fruit_strawberry}")
print(f"Cosine similarity between 'Strawberry' and 'Fruit': {similarity_fruit}")
print(f"Cosine similarity between 'Strawberry' and 'Season': {similarity_season}")
print(f"Cosine similarity between 'Strawberry' and 'Sport': {similarity_sport}")
print(f"Cosine similarity between 'Strawberry' and 'Cuisine': {similarity_cuisine}")

print(f"Cosine similarity between 'Kiwi' and 'Fruit': {similarity_fruit_kiwi}")
print(f"Cosine similarity between 'Kiwi' and 'Season': {similarity_season_kiwi}")
print(f"Cosine similarity between 'Kiwi' and 'Sport': {similarity_sport_kiwi}")
print(f"Cosine similarity between 'Kiwi' and 'Cuisine': {similarity_cuisine_kiwi}")

# represents 0 for fruits filled, 1 season filled, 0 for cuisines filled, 0 for sports filled
game_state = [0, 1, 0, 0]

# Hash the game state
hashed_value = hash_game_state(game_state)
print(f"Hashed value: {hashed_value}")

# Dehash the value back to the game state
dehashed_state = dehash_game_state(hashed_value)
print(f"Dehashed game state: {dehashed_state}")

# print(transition_matrix)