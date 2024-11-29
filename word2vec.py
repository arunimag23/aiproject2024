import gensim.downloader as api
import numpy as np

# Load pre-trained Word2Vec model (e.g., Google News)
model = api.load('word2vec-google-news-300')

# Define representative words for each category
category_representatives = {
    'Seasons': 'Summer',
    'Sports': 'Soccer',
    'Cuisines': 'Pizza',
    'Fruits': 'Apple'
}

def get_similarity(word, category_word):
    try:
        return model.similarity(word, category_word)
    except KeyError:
        # Handle words not in the vocabulary
        return 0.0

# Example of calculating similarities
word = 'Winter'
similarities = {category: get_similarity(word, representative) 
                for category, representative in category_representatives.items()}
print(similarities)


# Define the categories
categories = ['Seasons', 'Sports', 'Cuisines', 'Fruits']

# Calculate transition probabilities for a word
def get_transition_probabilities(word):
    similarities = {category: get_similarity(word, representative) 
                    for category, representative in category_representatives.items()}
    total_similarity = sum(similarities.values())
    # Normalize to get probabilities
    if total_similarity == 0:
        return {category: 0.25 for category in categories}  # Equal probability if no similarities found
    return {category: sim / total_similarity for category, sim in similarities.items()}

# Example: Get transition probabilities for a word
word = 'Winter'
transitions = get_transition_probabilities(word)
print("Transition probabilities:", transitions)

def classify_word(word):
    # Get transition probabilities
    probabilities = get_transition_probabilities(word)
    # Select category with the highest probability
    chosen_category = max(probabilities, key=probabilities.get)
    return chosen_category

# Example: Classify a word
word = 'Pizza'
predicted_category = classify_word(word)
print(f"The word '{word}' is predicted to belong to category '{predicted_category}'")