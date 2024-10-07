import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import ClassifierMixin

# Read the CSV files into Pandas Dataframes
squat_df = pd.read_csv("../Squat/squatdata.csv")
plank_df = pd.read_csv("../Plank/ds.csv")

# Drop the 'label' column from both DataFrames if it exists
if 'label' in squat_df.columns:
    squat_df.drop('label', axis=1, inplace=True)
if 'label' in plank_df.columns:
    plank_df.drop('label', axis=1, inplace=True)

# Sample 5000 rows from each DataFrame
squat_df_balanced = squat_df.sample(n=5000, random_state=42)
plank_df_balanced = plank_df.sample(n=5000, random_state=42)

# Identify common columns
common_columns = list(set(squat_df_balanced.columns) & set(plank_df_balanced.columns))

# Create a filtered DataFrame with only common columns
combined_df_filtered_balanced = pd.concat([squat_df_balanced, plank_df_balanced], ignore_index=True)[common_columns]

# Add 'source' column to indicate the origin of data
squat_df_balanced['source'] = 0
plank_df_balanced['source'] = 1

# Combine the balanced DataFrames again (after adding 'source')
combined_df_balanced = pd.concat([squat_df_balanced, plank_df_balanced], ignore_index=True)

# Add the 'source' column from combined_df_balanced to combined_df_filtered_balanced
combined_df_filtered_balanced['source'] = combined_df_balanced['source']

# Separate features and target
X = combined_df_filtered_balanced.drop('source', axis=1)
y = combined_df_balanced['source']

# Convert to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.long)

# Manual splitting
train_size = int(0.6 * len(combined_df_filtered_balanced))
val_size = int(0.2 * len(combined_df_filtered_balanced))
test_size = len(combined_df_filtered_balanced) - train_size - val_size

combined_df_filtered_balanced = combined_df_filtered_balanced.sample(frac=1, random_state=42) # Shuffle the dataframe

X = combined_df_filtered_balanced.drop('source', axis=1)
y = combined_df_filtered_balanced['source']

X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size+val_size], y[train_size+val_size:]

X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.long)

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print class distributions and unique values in 'source'
print("\nClass Distribution in Training Set:")
print(y_train.bincount())
print("\nUnique values in y_train:", torch.unique(y_train))

print("\nClass Distribution in Validation Set:")
print(y_val.bincount())
print("\nUnique values in y_val:", torch.unique(y_val))

print("\nClass Distribution in Testing Set:")
print(y_test.bincount())
print("\nUnique values in y_test:", torch.unique(y_test))

# Define the model architecture
class ExerciseClassifier(nn.Module, ClassifierMixin):  # Inherit from ClassifierMixin
    def __init__(self, input_size):
        super(ExerciseClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

    # Add fit method for sklearn compatibility
    def fit(self, X, y, epochs=100):
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Create DataLoader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Calculate class weights
        class_counts = y_tensor.bincount()
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.SGD(self.parameters(), lr=0.01)

        # Train the model
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        return self

    # Add predict_proba method for sklearn compatibility
    def predict_proba(self, X):
        with torch.no_grad():
            logits = self(torch.tensor(X, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)
            return probabilities.numpy()

    def predict(self, X):
        with torch.no_grad():
            logits = self(torch.tensor(X, dtype=torch.float32))
            _, predicted_classes = torch.max(logits, 1)  # Get predicted class labels
            return predicted_classes.numpy()

# Initialize the model
input_size = X_train.shape[1]
model = ExerciseClassifier(input_size)

# Train the model (using the fit method now)
model.fit(X_train.numpy(), y_train.numpy())

# Evaluate the model on the test set
y_pred_proba = model.predict_proba(X_test.numpy())
y_pred_classes = y_pred_proba.argmax(axis=1)
accuracy = (y_pred_classes == y_test.numpy()).mean()
print(f'\nFinal Accuracy on test set: {accuracy:.4f}')

# Compute and print confusion matrix
cm = confusion_matrix(y_test.numpy(), y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance Analysis (using permutation importance)
result = permutation_importance(model, X_test.numpy(), y_test.numpy(), n_repeats=10, random_state=42)

feature_importances = pd.Series(result.importances_mean, index=X.columns)

# Print feature importances
print("\nFeature Importances:")
print(feature_importances.sort_values(ascending=False))

# Visualize data distribution for a few important features
top_features = feature_importances.nlargest(3).index  # Select top 3 features

for feature in top_features:
    plt.figure()
    sns.histplot(X[y == 0][feature], color='blue', label='Squat', kde=True)
    sns.histplot(X[y == 1][feature], color='red', label='Plank', kde=True)
    plt.title(f"Distribution of {feature}")
    plt.legend()
    plt.show()