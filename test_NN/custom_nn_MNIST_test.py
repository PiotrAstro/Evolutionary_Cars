import numpy as np

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model

nn_structure = {
    "input_normal_size": 784,
    "out_actions_number": 10,
    "normal_hidden_layers": 2,
    "normal_hidden_neurons": 256,
    "dropout": 0.5,
    "normal_activation_function": "relu",  # "relu"
    "last_activation_function": [("softmax", 10)],
}
loss = [("Cross_Entropy", 10)]
batch_size = 256
lr = 0.01 #/ batch_size

labels_file_path = r"train_labels.txt"
data_file_path = r"train_correct.txt"
validation_num = 500
data_num = 10000


labels = np.loadtxt(labels_file_path).astype(np.int32)

permutation = np.random.permutation(len(labels))
labels = np.array(labels[permutation])

labels_one_hot = np.zeros((labels.size, labels.max() + 1))
labels_one_hot[np.arange(labels.size), labels] = 1

with open(data_file_path) as f:
    data_text = f.readlines()
data = np.array([np.fromstring(text, dtype=int, sep=',') for text in data_text])

data = np.array(data[permutation]) / 255

labels = labels[:data_num]
counted_labels = [np.sum(labels == i) for i in range(10)]
print(counted_labels)
labels_one_hot = labels_one_hot.astype(np.float32)[:data_num]
data = data.astype(np.float32)[:data_num]

batch_test_labels = labels[:validation_num]
batch_test_data = data[:validation_num]

labels = labels[validation_num:]
labels_one_hot = labels_one_hot[validation_num:]
data = data[validation_num:]

neural_network = Normal_model(**nn_structure)
for generation in range(100):
    predictions = neural_network.p_forward_pass(batch_test_data)
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == batch_test_labels)
    # params = neural_network.get_parameters()
    print(f"Generation {generation}, accuracy: {accuracy}")

    for batch_start in range(0, len(data) - batch_size, batch_size):
        batch_data = data[batch_start:batch_start + batch_size]
        batch_labels = labels_one_hot[batch_start:batch_start + batch_size]
        neural_network.backward_SGD(batch_data, batch_labels, lr, loss)


