import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Question 7: Converting Text Data to Analyzable numerical data

# 1. Defining the Initial Table

# Define initial dataframe
data = [
    {"Movie": "Inception", "Genre": "Sci-Fi", "Duration (min)": 148, "Rating (1-10)": 8.8, "HasAwards": "Yes"},
    {"Movie": "Toy Story", "Genre": "Animation", "Duration (min)": 81, "Rating (1-10)": 8.3, "HasAwards": "Yes"},
    {"Movie": "Fast & Furious", "Genre": "Action", "Duration (min)": 130, "Rating (1-10)": 6.9, "HasAwards": "No"},
]

df = pd.DataFrame(data)
print("\nInitial DataFrame:\n")
print(df)
print("\n" + "="*50 + "\n")

# 2. Adding 7 New Movies

# Add 7 new movies
new_movies = [
    {"Movie": "The Godfather", "Genre": "Crime", "Duration (min)": 175, "Rating (1-10)": 9.2, "HasAwards": "Yes"},
    {"Movie": "The Dark Knight", "Genre": "Action", "Duration (min)": 152, "Rating (1-10)": 9.0, "HasAwards": "Yes"},
    {"Movie": "Pulp Fiction", "Genre": "Crime", "Duration (min)": 154, "Rating (1-10)": 8.9, "HasAwards": "Yes"},
    {"Movie": "Forrest Gump", "Genre": "Drama", "Duration (min)": 142, "Rating (1-10)": 8.8, "HasAwards": "Yes"},
    {"Movie": "The Matrix", "Genre": "Sci-Fi", "Duration (min)": 136, "Rating (1-10)": 8.7, "HasAwards": "Yes"},
    {"Movie": "Goodfellas", "Genre": "Crime", "Duration (min)": 146, "Rating (1-10)": 8.7, "HasAwards": "Yes"},
    {"Movie": "The Avengers", "Genre": "Action", "Duration (min)": 143, "Rating (1-10)": 8.0, "HasAwards": "No"},
]

# Add new movies to dataframe
df = pd.concat([df, pd.DataFrame(new_movies)], ignore_index=True)

print("DataFrame after adding new movies:")
print(df)
print("\n" + "="*50 + "\n")

# 3. Converting HasAwards Column to Numeric

# Convert Yes → 1 , No → 0
df['HasAwards'] = df['HasAwards'].map({'Yes': 1, 'No': 0})

print("DataFrame after converting HasAwards:")
print(df)
print("\n" + "="*50 + "\n")

# 4. Converting Genre Column with Two Methods

# Method 1: Label Encoding

# Label Encoding for Genre

# Create encoder
label_encoder = LabelEncoder()

# Create copy of dataframe for Label Encoding
df_labeled = df.copy()
df_labeled['Genre_Label'] = label_encoder.fit_transform(df_labeled['Genre'])

print("Label Encoding Results:")
print("Genre Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print(df_labeled[['Movie', 'Genre', 'Genre_Label']])
print("\n" + "="*50 + "\n")

# Method 2: One-Hot Encoding

# One-Hot Encoding for Genre
df_onehot = df.copy()
genre_dummies = pd.get_dummies(df_onehot['Genre'], prefix='Genre')

# Add One-Hot columns to dataframe
df_onehot = pd.concat([df_onehot, genre_dummies], axis=1)

print("One-Hot Encoding Results:")
print(df_onehot[['Movie', 'Genre'] + list(genre_dummies.columns)])
print("\n" + "="*50 + "\n")

################################################################################

# Question 8: Conditional Feature - Creating New Column with if/else

# Creating LikelyToWinAward column based on conditions
df['LikelyToWinAward'] = 0  # Initialize with 0

# Apply condition: Rating > 8 AND Duration > 100
condition = (df['Rating (1-10)'] > 8) & (df['Duration (min)'] > 100)
df.loc[condition, 'LikelyToWinAward'] = 1

print("DataFrame with LikelyToWinAward column:")
print(df[['Movie', 'Rating (1-10)', 'Duration (min)', 'HasAwards', 'LikelyToWinAward']])
print("\n" + "="*50 + "\n")

# Alternative method using numpy.where
df['LikelyToWinAward_Alt'] = np.where((df['Rating (1-10)'] > 8) & (df['Duration (min)'] > 100), 1, 0)

print("Alternative method using numpy.where:")
print(df[['Movie', 'Rating (1-10)', 'Duration (min)', 'HasAwards', 'LikelyToWinAward', 'LikelyToWinAward_Alt']])
print("\n" + "="*50 + "\n")

# Analysis of the new feature
print("Analysis of LikelyToWinAward feature:")
print(f"Total movies likely to win award: {df['LikelyToWinAward'].sum()}")
print(f"Percentage of movies likely to win award: {(df['LikelyToWinAward'].sum() / len(df) * 100):.1f}%")

# Compare with actual awards
correct_predictions = (df['LikelyToWinAward'] == df['HasAwards']).sum()
accuracy = correct_predictions / len(df) * 100
print(f"Accuracy compared to actual awards: {accuracy:.1f}%")

# Movies that are misclassified
misclassified = df[df['LikelyToWinAward'] != df['HasAwards']]
print(f"\nMisclassified movies ({len(misclassified)}):")
print(misclassified[['Movie', 'Rating (1-10)', 'Duration (min)', 'HasAwards', 'LikelyToWinAward']])

print("\n" + "="*80 + "\n")

# Final Results And Analysis

# Display final dataframe with both methods
print("Final DataFrame with Label Encoding and LikelyToWinAward:")
df_labeled['LikelyToWinAward'] = df['LikelyToWinAward']
print(df_labeled)

print("\n" + "="*80 + "\n")

print("Final DataFrame with One-Hot Encoding and LikelyToWinAward:")
df_onehot['LikelyToWinAward'] = df['LikelyToWinAward']
print(df_onehot)

print("\n" + "="*80 + "\n")

# General information about the dataframe
print("DataFrame Overview:")
print(f"Total movies: {len(df)}")
print(f"Unique genres: {df['Genre'].unique()}")
print(f"Movies with awards: {df['HasAwards'].sum()}")
print(f"Movies likely to win awards: {df['LikelyToWinAward'].sum()}")
print(f"Average rating: {df['Rating (1-10)'].mean():.2f}")
print(f"Average duration: {df['Duration (min)'].mean():.1f} minutes")