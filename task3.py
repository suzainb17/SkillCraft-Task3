import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to save the plot as image
import matplotlib.pyplot as plt

print("Step 1: Loading local dataset...")
# Load dataset from local file
data = pd.read_csv("bank.csv", sep=';')
print("Dataset loaded successfully!")

print("Step 2: Converting categorical columns to numeric...")
data = pd.get_dummies(data, drop_first=True)

print("Step 3: Splitting features and target...")
X = data.drop('y_yes', axis=1)
y = data['y_yes']

print("Step 4: Splitting into training and testing data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 5: Building Decision Tree model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("Step 6: Making predictions...")
y_pred = model.predict(X_test)

print("\nStep 7: Model Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

print("Step 8: Saving decision tree image...")
plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], fontsize=8)
plt.savefig("decision_tree.png")  # Saves image in the project folder
print("Decision tree saved as 'decision_tree.png' in your folder!")