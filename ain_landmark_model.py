import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib


df = pd.read_csv("isl_landmarks_2hands.csv")
X = df.drop('label', axis=1).values
y = df['label'].astype(str).values


le = LabelEncoder()
y_encoded = le.fit_transform(y)
joblib.dump(le, "label_encoder_2hands.pkl")


k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []
models = []

print("Starting Stratified K-Fold Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded)):
    print(f"\n🔧 Fold {fold+1}/{k}...")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    accuracies.append(acc)
    models.append(clf)

    print(f" Fold {fold+1} Accuracy: {acc:.4f}")


best_index = np.argmax(accuracies)
best_model = models[best_index]
joblib.dump(best_model, "isl_landmark_model_2hands.pkl")

print(f"\n Best model from Fold {best_index+1} with Accuracy: {accuracies[best_index]:.4f}")
print("Model saved as isl_landmark_model_2hands.pkl")
