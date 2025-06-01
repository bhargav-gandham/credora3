# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load Data
df = pd.read_csv('bank-full.csv', sep=';')  # Adjust sep if needed
print("First few rows:")
print(df.head())

# 3. Data Preprocessing
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# If missing values exist, handle them (here, let's just drop for example)
df = df.dropna()

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# 4. Feature Selection (using all except target here)
X = df.drop('y', axis=1)  # 'y' is the target
y = df['y']

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Decision Tree Model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

# 7. Evaluate
y_pred = dtree.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Visualize the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=label_encoders['y'].classes_, filled=True, max_depth=3)
plt.title("Decision Tree Visualization (Top 3 Levels)")
plt.show()