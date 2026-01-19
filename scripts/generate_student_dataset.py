"""
Generate Student Degree Classification Dataset

Creates a CSV dataset with student features and degree classifications.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Create data directory
data_dir = Path('data/neural_networks')
data_dir.mkdir(parents=True, exist_ok=True)

print("Generating Student Degree Classification Dataset...")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n_students = 1000

# Generate features
# Features that might affect student performance
attendance = np.random.uniform(60, 100, n_students)  # 60-100%
quiz_avg = np.random.uniform(40, 95, n_students)  # 40-95
assignment_avg = np.random.uniform(45, 98, n_students)  # 45-98
midterm_score = np.random.uniform(35, 100, n_students)  # 35-100
project_score = np.random.uniform(50, 100, n_students)  # 50-100
study_hours_per_week = np.random.uniform(5, 40, n_students)  # 5-40 hours
participation_score = np.random.uniform(30, 100, n_students)  # 30-100

# Calculate final score (weighted average)
# Realistic weights for different components
final_score = (
    0.10 * attendance +
    0.15 * quiz_avg +
    0.20 * assignment_avg +
    0.25 * midterm_score +
    0.20 * project_score +
    0.05 * participation_score +
    0.05 * (study_hours_per_week / 40 * 100)  # Normalize study hours
)

# Add some noise
final_score += np.random.normal(0, 3, n_students)
final_score = np.clip(final_score, 0, 100)  # Ensure score is between 0-100

# Classify into degree categories
def classify_degree(score):
    """Classify score into degree category"""
    if score < 50:
        return 'Bad'
    elif score < 65:
        return 'Acceptable'
    elif score < 75:
        return 'Good'
    elif score < 85:
        return 'Very Good'
    else:
        return 'Excellent'

degree_category = [classify_degree(score) for score in final_score]

# Create DataFrame
df = pd.DataFrame({
    'student_id': range(1, n_students + 1),
    'attendance': np.round(attendance, 2),
    'quiz_avg': np.round(quiz_avg, 2),
    'assignment_avg': np.round(assignment_avg, 2),
    'midterm_score': np.round(midterm_score, 2),
    'project_score': np.round(project_score, 2),
    'study_hours_per_week': np.round(study_hours_per_week, 1),
    'participation_score': np.round(participation_score, 2),
    'final_score': np.round(final_score, 2),
    'degree_category': degree_category
})

# Save to CSV
csv_path = data_dir / 'student_degree_dataset.csv'
df.to_csv(csv_path, index=False)

print(f"\n[OK] Dataset generated: {n_students} students")
print(f"    Saved to: {csv_path}")
print(f"\nDataset Statistics:")
print(f"  Features: {len(df.columns) - 3}")  # Exclude student_id, final_score, degree_category
print(f"  Degree Distribution:")
print(df['degree_category'].value_counts().sort_index())
print(f"\nScore Statistics:")
print(f"  Mean: {final_score.mean():.2f}")
print(f"  Std: {final_score.std():.2f}")
print(f"  Min: {final_score.min():.2f}")
print(f"  Max: {final_score.max():.2f}")

# Also save as numpy arrays for easy loading
X = df[['attendance', 'quiz_avg', 'assignment_avg', 'midterm_score', 
        'project_score', 'study_hours_per_week', 'participation_score']].values
y_scores = df['final_score'].values
y_categories = df['degree_category'].values

# One-hot encode categories
category_mapping = {'Bad': 0, 'Acceptable': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4}
y_encoded = np.array([category_mapping[cat] for cat in y_categories])

# Save numpy arrays
np.save(data_dir / 'student_X.npy', X.astype(np.float32))
np.save(data_dir / 'student_y_scores.npy', y_scores.astype(np.float32))
np.save(data_dir / 'student_y_categories.npy', y_categories)
np.save(data_dir / 'student_y_encoded.npy', y_encoded)

print(f"\n[OK] NumPy arrays saved for easy loading")
print(f"    X shape: {X.shape}")
print(f"    y_scores shape: {y_scores.shape}")
print(f"    y_categories shape: {y_categories.shape}")

print("\n" + "=" * 60)
print("Dataset generation complete!")
print("=" * 60)
