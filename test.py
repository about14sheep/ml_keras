from keras.datasets import imbd

(train_data, train_labels), (test_data, test_labels) = imbd.load_data(num_words=10000)
print(train_data[0])
