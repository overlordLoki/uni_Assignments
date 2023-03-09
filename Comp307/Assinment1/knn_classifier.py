import numpy as np
import sys

## Define the min-max normalization function
def min_max_normalization(X):
    X = np.array(X)
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return (X - X_min) / (X_max - X_min)

## Define the KNN classifier function
def knn_classifier(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(len(X_test)):
        distances = np.sqrt(np.sum((X_train - X_test[i])**2, axis=1))
        nearest_neighbors = np.argsort(distances)[:k]
        nearest_classes = y_train[nearest_neighbors]
        classes, counts = np.unique(nearest_classes, return_counts=True)
        y_pred.append(classes[np.argmax(counts)])
    return y_pred

## Read the training and testing dataset
def read_data(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f, delimiter=' ')
        X = data[:, :-1]
        y = data[:, -1].astype(int)
    return X, y

## Define the main function
def main():
    
    ## Get the command line arguments
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]

    ## Read the training and testing data
    X_train, y_train = read_data(train_filename)
    X_test, y_test = read_data(test_filename)
    
    ## Normalize the data using min-max normalization
    X_train = min_max_normalization(X_train)
    X_test = min_max_normalization(X_test)
    
    ## Classify the test set using KNN with k=1
    y_pred_1 = knn_classifier(X_train, y_train, X_test, 1)
    accuracy_1 = np.mean(y_pred_1 == y_test) * 100
    print("Accuracy of k-NN with k=1:", accuracy_1)
    
    ## Classify the test set using KNN with k=3
    y_pred_3 = knn_classifier(X_train, y_train, X_test, 3)
    accuracy_3 = np.mean(y_pred_3 == y_test) * 100
    print("Accuracy of k-NN with k=3:", accuracy_3)

## Call the main function
if __name__ == '__main__':
    main()

# Import numpy library
import numpy as np

# Define a function to calculate euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Define a function to get the k nearest neighbors
def get_knn(data, query, k):
    # Initialize an empty list to store distances and labels
    distances = []
    
    # Loop through each data point
    for i in range(len(data)):
        # Get the features and label of the current point
        features = data[i][:-1]
        label = data[i][-1]
        
        # Calculate the distance between the query and the current point
        distance = euclidean_distance(features, query)
        
        # Append the distance and label to the list
        distances.append((distance, label))
    
    # Sort the list by distance in ascending order
    distances.sort()
    
    # Get the k nearest neighbors
    neighbors = distances[:k]
    
    return neighbors

# Define a function to predict the label of a query based on majority voting
def predict_label(neighbors):
    # Initialize a dictionary to store votes for each label
    votes = {}
    
    # Loop through each neighbor
    for neighbor in neighbors:
        # Get the label of the neighbor
        label = neighbor[1]
        
        # Increment the vote count for that label or set it to 1 if not present
        if label in votes:
            votes[label] += 1
        else:
            votes[label] = 1
    
    # Get the label with maximum votes 
    max_votes = max(votes.values())
    
    # Loop through each vote and check if it is equal to max_votes 
    for vote in votes.items():
        if vote[1] == max_votes:
            # Return that label as prediction 
            return vote[0]