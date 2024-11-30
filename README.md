# Overview
This project develops an agent to tackle a unique variation of the New York Times' Connections game, where 16 words must be grouped into 4 categories. The agent employs a Hidden Markov Model and leverages Google's pretrained Word2Vec model to accurately classify words into their respective categories.

# Features
**Analyzing Word Semantic Similarity**

This process leverages Google's Word2Vec model to calculate cosine similarity between each word and the four predefined categories. The similarities are used to construct an emission matrix, which is combined with a transition matrix. The Viterbi algorithm is then applied to efficiently group words into their corresponding categories. A colored board is returned that showacases the grid of words colored in accordance to its matching category.

# Document Overview
**TO DO**

# Getting Started
Prerequisites
Ensure you have the following installed to run the front end application

- Python 3.7+
- numpy < 2
- pygame
- Word2Vec
- To download Google's Word2Vec model, use this [Link](https://drive.google.com/file/d/1ETEzH8X7uM_xXtIEuNLgz9VL7eQEeE_V/view)
- scikit-learn

# Installation Instructions 
Clone the repository:
`git clone (https://github.com/arunimag23/aiproject2024.git)`

Navigate to the project directory:
cd 

Install gensim:
`pip install gensim`

Install scikit-learn:
`pip install scikit-learn`

Run pygame front-end:

