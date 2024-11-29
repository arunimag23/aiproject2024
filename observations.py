import numpy as np

def word2vec_calculation(word1, word2):
    return

observations = np.array(["Strawberry", "Kiwi", "Spring", "Summer", 
                         "Tennis", "Fall", "Winter", "Soccer",
                         "Indian", "Thai", "Basketball", "Japanese",
                         "Fall", "Apple", "Chinese", "Banana"])

categories = np.array(["Fruits", "Seasons", "Sports", "Cuisines"])


states = np.array(["Fruit1", "Fruit2", "Fruit3", "Fruit4", 
                   "Season1", "Season2", "Season3", "Season4",
                   "Sport1", "Sport2", "Sport3", "Sport4",
                   "Cuisine1", "Cuisine2", "Cuisine3", "Cuisine4"])

# state to category mapping
category_state_mapping = {
    "Fruits": states[:4],
    "Seasons": states[4:8],
    "Sports": states[8:12],
    "Cuisines": states[12:]
}

#emission matrix of mappings for categories to states
emission_matrix_categories_states = np.zeros((len(categories), len(states)))

for category_index in range(len(categories)):
    start_idx = category_index * 4
    end_idx = start_idx + 4
    for state_index in range(start_idx, end_idx):
        emission_matrix_categories_states[category_index, state_index] = 1

#emission matrix of mappings for observations to states (placeholder, should change for word2vec mapping?)
emission_matrix_observations_states = np.zeros((len(observations), len(states)))


for observation_index in range(len(observations)):
    for state_index in range(len(states)):
        emission_matrix_observations_states[observation_index, state_index] = word2vec_calculation(observations[observation_index], states[state_index]) # replace with word2vec calculation of observation[observation_index] and states[state_index] 

#TO DO: create transition matrix

print("Emission Matrix (Categories x States):")
print(emission_matrix_categories_states)
#print(emission_matrix_observations_states)