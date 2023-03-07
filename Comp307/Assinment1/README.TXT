To run the program, open the command prompt and navigate to the directory containing the python file. 
Then type the following command:

python knn_classifier.py train_filename test_filename

where 'train_filename' is the name of the training file and 'test_filename' is the name of the test file.

For example:

python knn_classifier.py wine-training.csv wine-test.csv

The program will classify each instance in the test set wine-test according to the training set wine-training.
It will then print the classification accuracy on the test set for k-NN with k=1 and k=3.