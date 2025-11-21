import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

###################################################################################################
# Question 10: Award Prediction Visualization
print("\n" + "=" * 80)
print("QUESTION 10: Award Prediction Visualization")
print("=" * 80)

# Create visualization figure
# plt.figure(figsize=(15, 10))
plt.figure(figsize=(12, 6))

# 1. Bar chart comparing Prediction vs Actual Awards
plt.subplot(2, 2, 1)

# Prepare data for comparison
comparison_data = pd.DataFrame({
    'Category': ['Predicted to Win', 'Predicted to Win', 'Predicted Not to Win', 'Predicted Not to Win',
                 'Actual Winners', 'Actual Winners', 'Actual Non-Winners', 'Actual Non-Winners'],
    'Status': ['Correct', 'Incorrect', 'Correct', 'Incorrect', 'Has Awards', 'No Awards', 'Has Awards', 'No Awards'],
    'Count': [
        true_positives, false_positives,  # Predicted to Win
        true_negatives, false_negatives,  # Predicted Not to Win
        (df['HasAwards'] == 1).sum(), (df['HasAwards'] == 0).sum(),  # Actual
        (df['HasAwards'] == 1).sum(), (df['HasAwards'] == 0).sum()   # Actual
    ]
})

# Filter for prediction comparison
prediction_comparison = comparison_data[comparison_data['Category'].isin(['Predicted to Win', 'Predicted Not to Win'])]

sns.barplot(data=prediction_comparison, x='Category', y='Count', hue='Status', 
            palette={'Correct': 'green', 'Incorrect': 'red'})
plt.title('Prediction vs Actual Awards Comparison', fontsize=10, fontweight='bold')
plt.xlabel('Prediction Category')
plt.ylabel('Number of Movies')
plt.legend(title='Prediction Status')
plt.grid(axis='y', alpha=0.3)

# 2. Countplot for LikelyToWinAward vs HasAwards
plt.subplot(2, 2, 2)

# Create a cross-tabulation for the heatmap style countplot
cross_tab = pd.crosstab(df['HasAwards'], df['LikelyToWinAward'])
cross_tab.index = ['No Award', 'Has Award']
cross_tab.columns = ['Predicted No', 'Predicted Yes']

sns.heatmap(cross_tab, annot=True, fmt='d', cmap='YlOrRd', cbar=True, 
            square=True, linewidths=0.5, linecolor='black')
plt.title('Prediction vs Actual Awards Heatmap', fontsize=10, fontweight='bold')
plt.xlabel('Predicted Award')
plt.ylabel('Actual Award')

# 3. Side-by-side comparison for each movie
plt.subplot(2, 2, 3)
movies = df['Movie']
x = np.arange(len(movies))
width = 0.35

plt.bar(x - width/2, df['HasAwards'], width, label='Actual Awards', color='blue', alpha=0.7)
plt.bar(x + width/2, df['LikelyToWinAward'], width, label='Predicted Awards', color='orange', alpha=0.7)

plt.xlabel('Movies')
plt.ylabel('Award Status (0=No, 1=Yes)')
plt.title('Actual vs Predicted Awards for Each Movie', fontsize=10, fontweight='bold')
plt.xticks(x, movies, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 4. Performance metrics visualization
plt.subplot(2, 2, 4)
metrics = ['Accuracy', 'Precision', 'Recall']
values = [accuracy, precision, recall]
colors = ['green', 'blue', 'orange']

bars = plt.bar(metrics, values, color=colors, alpha=0.7)
plt.title('Prediction Performance Metrics', fontsize=10, fontweight='bold')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# figure 2 

# Create a figure with multiple subplots
# fig = plt.figure(figsize=(18, 12))
fig = plt.figure(figsize=(11, 7))

# 1. Confusion Matrix Heatmap
plt.subplot(2, 3, 1)
confusion_matrix = pd.crosstab(df['HasAwards'], df['LikelyToWinAward'], 
                              rownames=['Actual Awards'], 
                              colnames=['Predicted Awards'])

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix: Actual vs Predicted Awards', fontsize=8, fontweight='bold')

# 2. Countplot comparing Actual vs Predicted
plt.subplot(2, 3, 2)
comparison_df = pd.DataFrame({
    'Type': ['Actual Awards', 'Actual Awards', 'Predicted Awards', 'Predicted Awards'],
    'Value': [0, 1, 0, 1],
    'Count': [
        (df['HasAwards'] == 0).sum(), (df['HasAwards'] == 1).sum(),
        (df['LikelyToWinAward'] == 0).sum(), (df['LikelyToWinAward'] == 1).sum()
    ]
})

sns.barplot(data=comparison_df, x='Type', y='Count', hue='Value', palette=['red', 'green'])
plt.title('Actual Awards vs Predicted Awards Distribution', fontsize=8, fontweight='bold')
plt.legend(title='Award Status', labels=['No Award', 'Has Award'])

# 3. Side-by-side comparison
plt.subplot(2, 3, 3)
comparison_data = pd.DataFrame({
    'Movie': df['Movie'],
    'Actual': df['HasAwards'],
    'Predicted': df['LikelyToWinAward'],
    'Match': df['HasAwards'] == df['LikelyToWinAward']
})

# Create grouped bar chart
x_pos = np.arange(len(comparison_data))
width = 0.35

plt.bar(x_pos - width/2, comparison_data['Actual'], width, label='Actual Awards', color='blue', alpha=0.7)
plt.bar(x_pos + width/2, comparison_data['Predicted'], width, label='Predicted Awards', color='orange', alpha=0.7)

plt.xlabel('Movies')
plt.ylabel('Award Status (0=No, 1=Yes)')
plt.title('Actual vs Predicted Awards for Each Movie', fontsize=8, fontweight='bold')
plt.xticks(x_pos, comparison_data['Movie'], rotation=45, ha='right')
plt.legend()

# 4. Scatter plot showing Rating vs Duration with prediction results
plt.subplot(2, 3, 4)
colors = ['red' if actual != predicted else 'green' 
          for actual, predicted in zip(df['HasAwards'], df['LikelyToWinAward'])]
markers = ['o' if award == 1 else 's' for award in df['HasAwards']]

for i, (_, row) in enumerate(df.iterrows()):
    color = 'red' if row['HasAwards'] != row['LikelyToWinAward'] else 'green'
    marker = 'o' if row['HasAwards'] == 1 else 's'
    plt.scatter(row['Rating (1-10)'], row['Duration (min)'], 
               c=color, marker=marker, s=100, alpha=0.7)
    plt.annotate(row['Movie'], (row['Rating (1-10)'], row['Duration (min)']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

# Add decision boundary
plt.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Duration Threshold (100 min)')
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.5, label='Rating Threshold (8)')
plt.xlabel('Rating (1-10)')
plt.ylabel('Duration (min)')
plt.title('Rating vs Duration with Prediction Results\n(Green=Correct, Red=Incorrect)', fontsize=8, fontweight='bold')
plt.legend()

# 5. Accuracy by Genre
plt.subplot(2, 3, 5)
genre_accuracy = []
genres = df['Genre'].unique()

for genre in genres:
    genre_df = df[df['Genre'] == genre]
    if len(genre_df) > 0:
        accuracy = (genre_df['HasAwards'] == genre_df['LikelyToWinAward']).mean() * 100
        genre_accuracy.append((genre, accuracy, len(genre_df)))

genre_accuracy_df = pd.DataFrame(genre_accuracy, columns=['Genre', 'Accuracy', 'Count'])
sns.barplot(data=genre_accuracy_df, x='Genre', y='Accuracy', palette='viridis')
plt.title('Prediction Accuracy by Genre', fontsize=8, fontweight='bold')
plt.xticks(rotation=45)
plt.ylabel('Accuracy (%)')

# 6. Prediction Performance Metrics
plt.subplot(2, 3, 6)
metrics = ['Accuracy', 'Precision', 'Recall']
true_positives = ((df['LikelyToWinAward'] == 1) & (df['HasAwards'] == 1)).sum()
true_negatives = ((df['LikelyToWinAward'] == 0) & (df['HasAwards'] == 0)).sum()
false_positives = ((df['LikelyToWinAward'] == 1) & (df['HasAwards'] == 0)).sum()
false_negatives = ((df['LikelyToWinAward'] == 0) & (df['HasAwards'] == 1)).sum()

accuracy = (true_positives + true_negatives) / len(df) * 100
precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0

metric_values = [accuracy, precision, recall]
colors = ['green', 'blue', 'orange']

bars = plt.bar(metrics, metric_values, color=colors, alpha=0.7)
plt.title('Prediction Performance Metrics', fontsize=8, fontweight='bold')
plt.ylabel('Percentage (%)')

# Add value labels on bars
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

###################################################################################################
# Analysis and Conclusions for Question 10
print("\n" + "=" * 80)
print("VISUALIZATION ANALYSIS AND CONCLUSIONS")
print("=" * 80)

# 1. Are predictions aligned with reality?
print("\n1. PREDICTION ALIGNMENT WITH REALITY:")
print("-" * 40)

print(f"Overall Accuracy: {accuracy:.1f}%")
print(f"Correct Predictions: {total_correct_predictions}/{total_movies}")

if accuracy >= 80:
    alignment_status = "EXCELLENT"
    alignment_color = "🟢"
elif accuracy >= 70:
    alignment_status = "GOOD" 
    alignment_color = "🟡"
elif accuracy >= 60:
    alignment_status = "MODERATE"
    alignment_color = "🟠"
else:
    alignment_status = "POOR"
    alignment_color = "🔴"

print(f"Alignment Status: {alignment_color} {alignment_status}")
print(f"The predictions show {alignment_status.lower()} alignment with reality.")

""" way 2:
correct_predictions = (df['HasAwards'] == df['LikelyToWinAward']).sum()
accuracy = correct_predictions / len(df) * 100

print(f"Total correct predictions: {correct_predictions}/{len(df)} ({accuracy:.1f}%)")
print(f"Prediction alignment: {'GOOD' if accuracy >= 70 else 'MODERATE' if accuracy >= 50 else 'POOR'}")
"""

# 2. Which movies were incorrectly predicted?
print("\n2. INCORRECTLY PREDICTED MOVIES ANALYSIS:")
print("-" * 40)

incorrect_predictions = df[df['HasAwards'] != df['LikelyToWinAward']]

if len(incorrect_predictions) > 0:
    print(f"Total incorrect predictions: {len(incorrect_predictions)} movies")
    
    # False Positives
    false_pos = incorrect_predictions[incorrect_predictions['LikelyToWinAward'] == 1]
    if len(false_pos) > 0:
        print(f"\nFalse Positives ({len(false_pos)} movies):")
        print("Movies predicted to win but didn't actually win:")
        for _, movie in false_pos.iterrows():
            print(f"  - {movie['Movie']} (Rating: {movie['Rating (1-10)']}, Duration: {movie['Duration (min)']}min)")
    
    # False Negatives
    false_neg = incorrect_predictions[incorrect_predictions['LikelyToWinAward'] == 0]
    if len(false_neg) > 0:
        print(f"\nFalse Negatives ({len(false_neg)} movies):")
        print("Movies predicted not to win but actually won:")
        for _, movie in false_neg.iterrows():
            print(f"  - {movie['Movie']} (Rating: {movie['Rating (1-10)']}, Duration: {movie['Duration (min)']}min)")
else:
    print("🎉 All predictions were correct! No incorrectly predicted movies.")

""" way 2:
if len(incorrect_predictions) > 0:
    print(f"Number of incorrect predictions: {len(incorrect_predictions)}")
    print("\nDetails of incorrect predictions:")
    incorrect_details = incorrect_predictions[['Movie', 'Genre', 'Rating (1-10)', 'Duration (min)', 'HasAwards', 'LikelyToWinAward']].copy()
    incorrect_details['Error Type'] = np.where(
        (incorrect_details['HasAwards'] == 0) & (incorrect_details['LikelyToWinAward'] == 1),
        'False Positive',
        'False Negative'
    )
    print(incorrect_details.to_string(index=False))
else:
    print("All predictions are correct!")
"""

# 3. Can changing prediction conditions improve accuracy?
print("\n3. PREDICTION RULE OPTIMIZATION ANALYSIS:")
print("-" * 40)

print("Current rule: Rating > 8 AND Duration > 100")
print(f"Current accuracy: {accuracy:.1f}%")

# Test alternative rules
alternative_rules = [
    ("Rating > 8.2 AND Duration > 100", (df['Rating (1-10)'] > 8.2) & (df['Duration (min)'] > 100)),
    ("Rating > 8 AND Duration > 120", (df['Rating (1-10)'] > 8) & (df['Duration (min)'] > 120)),
    ("Rating > 8.5", (df['Rating (1-10)'] > 8.5)),
    ("Rating > 8.2 AND Duration > 90", (df['Rating (1-10)'] > 8.2) & (df['Duration (min)'] > 90)),
    ("Rating > 8.1 AND Duration > 110", (df['Rating (1-10)'] > 8.1) & (df['Duration (min)'] > 110)),
]

print("\nTesting alternative prediction rules:")
best_rule = "Rating > 8 AND Duration > 100"
best_accuracy = accuracy

for rule_name, condition in alternative_rules:
    test_predictions = np.where(condition, 1, 0)
    test_accuracy = (test_predictions == df['HasAwards']).mean() * 100
    improvement = test_accuracy - accuracy
    
    improvement_symbol = "↑" if improvement > 0 else "↓" if improvement < 0 else "="
    print(f"  {rule_name:30} -> {test_accuracy:5.1f}% {improvement_symbol} {improvement:+.1f}%")
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_rule = rule_name

print(f"\nBest performing rule: {best_rule}")
print(f"Best achievable accuracy: {best_accuracy:.1f}%")

if best_accuracy > accuracy:
    improvement = best_accuracy - accuracy
    print(f"\u2705 Potential improvement: +{improvement:.1f}% by changing the rule")
    print("Recommendation: Consider optimizing the prediction rule")
else:
    print("✅ Current rule appears to be optimal")
    print("Recommendation: Keep the current prediction rule")

# Final comprehensive analysis
print("\n" + "=" * 80)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("=" * 80)

print(f"📊 Dataset Overview:")
print(f"   • Total movies: {total_movies}")
print(f"   • Movies with awards: {(df['HasAwards'] == 1).sum()}")
print(f"   • Movies predicted to win: {movies_likely_to_win}")

print(f"\n🎯 Prediction Performance:")
print(f"   • Overall Accuracy: {accuracy:.1f}%")
print(f"   • Precision: {precision:.1f}% (how many predicted winners actually won)")
print(f"   • Recall: {recall:.1f}% (how many actual winners were predicted)")

print(f"\n🔍 Key Findings:")
print(f"   • Prediction alignment: {alignment_status}")
print(f"   • Incorrect predictions: {len(incorrect_predictions)} movies")
print(f"   • Rule optimization potential: {'Yes' if best_accuracy > accuracy else 'No'}")

print(f"\n💡 Recommendations:")
if accuracy >= 80:
    print("   • Current rule is highly effective - maintain as is")
elif accuracy >= 70:
    print("   • Rule performs well - minor optimizations possible")
else:
    print("   • Consider significant rule revisions for better accuracy")

if false_negatives > 0:
    print("   • Review false negatives as they represent missed opportunities")
if false_positives > 0:
    print("   • Review false positives to reduce incorrect optimistic predictions")

# Additional insights
print("\n4. ADDITIONAL INSIGHTS:")
print("-" * 40)

# Analyze patterns in incorrect predictions

if len(incorrect_predictions) > 0:
    print("Patterns in incorrect predictions:")
    
    false_positives_df = incorrect_predictions[
        (incorrect_predictions['HasAwards'] == 0) & 
        (incorrect_predictions['LikelyToWinAward'] == 1)
    ]
    
    false_negatives_df = incorrect_predictions[
        (incorrect_predictions['HasAwards'] == 1) & 
        (incorrect_predictions['LikelyToWinAward'] == 0)
    ]
    
    if len(false_positives_df) > 0:
        print(f"False Positives (predicted to win but didn't): {len(false_positives_df)}")
        print("These movies met the criteria but didn't win awards")
        
    if len(false_negatives_df) > 0:
        print(f"False Negatives (predicted not to win but did): {len(false_negatives_df)}")
        print("These movies didn't meet criteria but won awards")
        
print(f"\n✅ Conclusion: The visualization clearly shows the relationship between")
print(f"   predicted and actual awards, highlighting areas for potential improvement")
print(f"   in the prediction rule for better alignment with reality.")
