import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

DATASET_FILE = 'dataset.csv'
MODEL_FILE = 'model.pkl'

if not os.path.exists(DATASET_FILE):
    print(f"Error: {DATASET_FILE} not found. Please run 1_collect_data.py first.")
    exit()

print("Loading dataset...")
df = pd.read_csv(DATASET_FILE)

if df.empty:
    print("Dataset is empty. Please collect data first.")
    exit()

# Features and Labels
X = df.drop('label', axis=1)
y = df['label']

# Split data (80/20)
print("Splitting data into 80% training and 20% testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
print("Training Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
print("Evaluating model...")
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
# average='weighted' to account for multi-class classification
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Save Confusion Matrix plot
print("\nGenerating Confusion Matrix image...")
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Saved as 'confusion_matrix.png'!")

# Save the trained model
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f)
    
print(f"\nModel successfully saved to {MODEL_FILE}")
