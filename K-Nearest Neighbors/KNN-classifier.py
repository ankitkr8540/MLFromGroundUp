import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Loading the training and testing dataset using pandas
training_dataset = pd.read_csv('training.csv')
testing_dataset = pd.read_csv('test.csv')

# Checking the first 5 rows of the training dataset
print(training_dataset.head())

# Define independent columns
independent_column = ['GDP', 'Inflation rate', 'Unemployment rate', "Daytime/evening attendance\t", 
                      'Displaced', 'Educational special needs', 'Debtor']

# Prepare training and testing data
X_train = training_dataset.drop(columns=independent_column + ['Target']).values
y_train = training_dataset['Target'].values
X_test = testing_dataset.drop(columns=independent_column).values

# Data Summary
print("Training dataset info:")
training_dataset.info()
print(f"Number of Duplicates: {training_dataset.duplicated().sum()}")

# Visualize outliers using a box plot
plt.figure(figsize=(20, 6))
training_dataset.boxplot()
plt.title("Box Plot of Numeric Features")
plt.xticks(rotation=45)
plt.savefig('boxplot.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(25, 25))
mask = np.triu(np.ones_like(training_dataset.corr(), dtype=bool))
heatmap = sns.heatmap(training_dataset.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize': 18}, pad=16)
plt.savefig('correlation_heatmap.png')
plt.close()

# Checking target distribution
print("Target distribution:")
print(training_dataset['Target'].value_counts())
print("\nCourse distribution:")
print(training_dataset['Course'].value_counts())

# Implementing the z-standardization to level on the same scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Below is the modified KNN algorithm with 1/d^2 weighting
class KNearestNeighborClass():
    def __init__(self, k):
        self.k = k

    def fitClassifier(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    # Modified to use 1/d^2 weighting
    def trainModel(self, X_test):
        y_pred = []
        for x in X_test:
            # Calculate distances
            distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            # Get corresponding labels and distances
            k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
            k_nearest_distances = [distances[i] for i in k_nearest_indices]
            
            # Handle special case where distance is zero (exact match)
            for i in range(len(k_nearest_distances)):
                if k_nearest_distances[i] == 0:
                    # If exact match found, use this label
                    predicted_label = k_nearest_labels[i]
                    break
            else:
                # Weighted voting using 1/d^2
                weights = [1 / (d ** 2) for d in k_nearest_distances]
                
                # Create a dictionary to store weighted votes for each class
                class_votes = {}
                for i, label in enumerate(k_nearest_labels):
                    if label not in class_votes:
                        class_votes[label] = 0
                    class_votes[label] += weights[i]
                
                # Get the class with the highest weighted vote
                predicted_label = max(class_votes, key=class_votes.get)
            
            y_pred.append(predicted_label)
        
        return np.array(y_pred)

# Below is the custom cross validation function
def custom_cross_val_score(X, y, k, n_folds=6):
    accuracies = []
    sizeOfFold = int(len(X)/n_folds)

    for i in range(n_folds):
        # Split the data into training and validation sets for this fold
        val_start = i * sizeOfFold
        val_end = (i + 1) * sizeOfFold
        X_validation_fold = X[val_start:val_end]
        y_validation_fold = y[val_start:val_end]
        X_train_fold = np.concatenate([X[:val_start], X[val_end:]])
        y_train_fold = np.concatenate([y[:val_start], y[val_end:]])

        # Creating an object of KNearestNeighborClass
        kNearestNeighborObject = KNearestNeighborClass(k=k)
        # Calling fitClassifier function using object
        kNearestNeighborObject.fitClassifier(X_train_fold, y_train_fold)

        # Make predictions on the validation set
        y_validation_prediction = kNearestNeighborObject.trainModel(X_validation_fold)

        # Calculate accuracy for this fold
        correct_predictions = 0
        for j in range(len(y_validation_fold)):
            if y_validation_fold[j] == y_validation_prediction[j]:
                correct_predictions += 1

        total_predictions = len(y_validation_fold)
        if total_predictions:
            fold_accuracy = (correct_predictions / total_predictions)
        else:
            fold_accuracy = 0
        accuracies.append(fold_accuracy)

    return np.mean(accuracies)

# Apply PCA to reduce dimensionality
n_components = 27  # You can adjust this number as needed
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Find the best k
best_k = None
best_accuracy = 0
print("Finding best k value...")
for k in range(5, 15):
    accuracy = custom_cross_val_score(X_train_pca, y_train, k, 5)
    print(f"k = {k}, Accuracy = {accuracy * 100:.2f}%")
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"\nBest k: {best_k}, Best Accuracy: {best_accuracy * 100:.2f}%")

# Train the KNN classifier with the best k
custom_knn = KNearestNeighborClass(k=best_k)
custom_knn.fitClassifier(X_train_pca, y_train)

# Make predictions on the test set
y_test_pred = custom_knn.trainModel(X_test_pca)

# Define the filename for the output text file
output_file = "testPredictionHW1Final.txt"

# Save predictions to file
with open(output_file, "w") as file:
    # Iterate through y_pred and write each prediction as a line in the file
    for prediction in y_test_pred:
        file.write(str(prediction) + "\n")

print(f"Predictions have been saved to {output_file}")