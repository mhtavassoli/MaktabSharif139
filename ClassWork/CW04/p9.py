import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Question 7: Converting Text Data to Analyzable numerical data

# 1. Defining the Initial Table
data = [
    {"Movie": "Inception", "Genre": "Sci-Fi", "Duration (min)": 148, "Rating (1-10)": 8.8, "HasAwards": "Yes"},
    {"Movie": "Toy Story", "Genre": "Animation", "Duration (min)": 81, "Rating (1-10)": 8.3, "HasAwards": "Yes"},
    {"Movie": "Fast & Furious", "Genre": "Action", "Duration (min)": 130, "Rating (1-10)": 6.9, "HasAwards": "No"},
]

df = pd.DataFrame(data)

# 2. Adding 7 New Movies
new_movies = [
    {"Movie": "The Godfather", "Genre": "Crime", "Duration (min)": 175, "Rating (1-10)": 9.2, "HasAwards": "Yes"},
    {"Movie": "The Dark Knight", "Genre": "Action", "Duration (min)": 152, "Rating (1-10)": 9.0, "HasAwards": "Yes"},
    {"Movie": "Pulp Fiction", "Genre": "Crime", "Duration (min)": 154, "Rating (1-10)": 8.9, "HasAwards": "Yes"},
    {"Movie": "Forrest Gump", "Genre": "Drama", "Duration (min)": 142, "Rating (1-10)": 8.8, "HasAwards": "Yes"},
    {"Movie": "The Matrix", "Genre": "Sci-Fi", "Duration (min)": 136, "Rating (1-10)": 8.7, "HasAwards": "Yes"},
    {"Movie": "Goodfellas", "Genre": "Crime", "Duration (min)": 146, "Rating (1-10)": 8.7, "HasAwards": "Yes"},
    {"Movie": "The Avengers", "Genre": "Action", "Duration (min)": 143, "Rating (1-10)": 8.0, "HasAwards": "No"},
]

df = pd.concat([df, pd.DataFrame(new_movies)], ignore_index=True)

# 3. Converting HasAwards Column to Numeric
df['HasAwards'] = df['HasAwards'].map({'Yes': 1, 'No': 0})

# 4. Converting Genre Column with Two Methods
# Label Encoding
label_encoder = LabelEncoder()
df_labeled = df.copy()
df_labeled['Genre_Label'] = label_encoder.fit_transform(df_labeled['Genre'])

# One-Hot Encoding
df_onehot = df.copy()
genre_dummies = pd.get_dummies(df_onehot['Genre'], prefix='Genre')
df_onehot = pd.concat([df_onehot, genre_dummies], axis=1)

###################################################################################################
# Question 8: Creating LikelyToWinAward column
df['LikelyToWinAward'] = np.where((df['Rating (1-10)'] > 8) & (df['Duration (min)'] > 100), 1, 0)

###################################################################################################
# Question 9: Analysis of Award Prediction
print("=" * 80)
print("QUESTION 9: Award Prediction Analysis with LikelyToWinAward Column")
print("=" * 80)

# 1. Number of movies that are "likely to win awards" according to the rule
movies_likely_to_win = df['LikelyToWinAward'].sum()
total_movies = len(df)

print("\n1. Analysis of movies likely to win awards:")
print("-" * 50)
print(f"Total movies: {total_movies}")
print(f"Number of movies 'likely to win awards': {movies_likely_to_win}")
print(f"Percentage of award-likely movies: {(movies_likely_to_win/total_movies)*100:.1f}%")

# Display movies likely to win awards
print("\nMovies likely to win awards (LikelyToWinAward = 1):")
likely_movies = df[df['LikelyToWinAward'] == 1][['Movie', 'Rating (1-10)', 'Duration (min)']]
print(likely_movies.to_string(index=False))

# 2. Checking prediction accuracy against actual HasAwards values
print("\n" + "=" * 80)
print("2. Prediction Accuracy Analysis:")
print("-" * 50)

# Create confusion matrix
true_positives = ((df['LikelyToWinAward'] == 1) & (df['HasAwards'] == 1)).sum()
true_negatives = ((df['LikelyToWinAward'] == 0) & (df['HasAwards'] == 0)).sum()
false_positives = ((df['LikelyToWinAward'] == 1) & (df['HasAwards'] == 0)).sum()
false_negatives = ((df['LikelyToWinAward'] == 0) & (df['HasAwards'] == 1)).sum()

print("Confusion Matrix:")
print(f"True Positives (TP): {true_positives} - Correct positive predictions")
print(f"True Negatives (TN): {true_negatives} - Correct negative predictions")
print(f"False Positives (FP): {false_positives} - Incorrect positive predictions")
print(f"False Negatives (FN): {false_negatives} - Incorrect negative predictions")

# Calculate accuracy metrics
total_correct_predictions = true_positives + true_negatives
accuracy = (total_correct_predictions / total_movies) * 100

precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0
recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0

print(f"\nPrediction Accuracy Metrics:")
print(f"Number of correct predictions: {total_correct_predictions} out of {total_movies}")
print(f"Overall Accuracy: {accuracy:.1f}%")
print(f"Precision: {precision:.1f}%")
print(f"Recall: {recall:.1f}%")

# Analysis of incorrect predictions
print("\n" + "=" * 80)
print("Analysis of Incorrect Predictions:")
print("-" * 50)

# False Positives (predicted to win award but didn't actually win)
false_positive_movies = df[(df['LikelyToWinAward'] == 1) & (df['HasAwards'] == 0)]
if len(false_positive_movies) > 0:
    print(f"\nMovies predicted to win awards but didn't actually win ({len(false_positive_movies)} movies):")
    print(false_positive_movies[['Movie', 'Rating (1-10)', 'Duration (min)', 'HasAwards', 'LikelyToWinAward']].to_string(index=False))
else:
    print("\nNo movies were predicted to win awards but didn't actually win.")

# False Negatives (predicted not to win award but actually won)
false_negative_movies = df[(df['LikelyToWinAward'] == 0) & (df['HasAwards'] == 1)]
if len(false_negative_movies) > 0:
    print(f"\nMovies predicted not to win awards but actually won ({len(false_negative_movies)} movies):")
    print(false_negative_movies[['Movie', 'Rating (1-10)', 'Duration (min)', 'HasAwards', 'LikelyToWinAward']].to_string(index=False))
else:
    print("\nNo movies were predicted not to win awards but actually won.")
############################################################################################################## 
# Advanced statistical analysis
print("\n" + "=" * 80)
print("Advanced Statistical Analysis:")
print("-" * 50)

# Average rating and duration for different groups
print("\nAverage rating and duration by award status:")
stats = df.groupby('HasAwards').agg({
    'Rating (1-10)': ['mean', 'std'],
    'Duration (min)': ['mean', 'std'],
    'Movie': 'count'
}).round(2)

print(stats)

print("\nAverage rating and duration by prediction status:")
prediction_stats = df.groupby('LikelyToWinAward').agg({
    'Rating (1-10)': ['mean', 'std'],
    'Duration (min)': ['mean', 'std'],
    'Movie': 'count'
}).round(2)

print(prediction_stats)

# Display final dataframe
print("\n" + "=" * 80)
print("Final DataFrame with All Columns:")
print("-" * 50)
final_df = df[['Movie', 'Genre', 'Duration (min)', 'Rating (1-10)', 'HasAwards', 'LikelyToWinAward']]
print(final_df.to_string(index=False))

print("\n" + "=" * 80)
print("Final Conclusion:")
print("-" * 50)
print(f"\u2705 The prediction rule (Rating > 8 AND Duration > 100) applies to {movies_likely_to_win} out of {total_movies} movies")
print(f"\u2705 {total_correct_predictions} predictions out of {total_movies} were correct ({accuracy:.1f}% accuracy)")
print(f"✅ This rule can be used as an initial criterion for identifying award-potential movies")

# Additional detailed analysis
print("\n" + "=" * 80)
print("Detailed Movie-by-Movie Analysis:")
print("-" * 50)
for index, row in df.iterrows():
    prediction_status = "CORRECT" if row['LikelyToWinAward'] == row['HasAwards'] else "INCORRECT"
    award_status = "Won Award" if row['HasAwards'] == 1 else "No Award"
    prediction_type = "Predicted to win" if row['LikelyToWinAward'] == 1 else "Predicted not to win"
    print(f"{row['Movie']}: {award_status} | {prediction_type} | {prediction_status}")

print("\n" + "=" * 80)
print("Rule Effectiveness Summary:")
print("-" * 50)
print(f"Rule: IF Rating > 8 AND Duration > 100 THEN LikelyToWinAward = 1")
print(f"Movies meeting criteria: {movies_likely_to_win}/{total_movies}")
print(f"Correct predictions: {total_correct_predictions}/{total_movies}")
print(f"Effectiveness: {accuracy:.1f}%")
