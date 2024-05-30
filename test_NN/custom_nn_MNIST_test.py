import numpy as np

from src_files.Neural_Network.Raw_Numpy.Raw_Numpy_Models.Normal.Normal_model import Normal_model

nn_structure = {
    "input_normal_size": 784,
    "out_actions_number": 10,
    "normal_hidden_layers": 2,
    "normal_hidden_neurons": 256,
    "normal_activation_function": "relu",  # "relu"
    "last_activation_function": [("softmax", 10)],
}
loss = [("Cross_Entropy", 10)]
lr = 0.01
batch_size = 32

labels_file_path = r"test_NN\train_labels.txt"
data_file_path = r"test_NN\train_correct.txt"



labels = np.loadtxt(labels_file_path).astype(np.int32)

permutation = np.random.permutation(len(labels))
labels = np.array(labels[permutation])

labels_one_hot = np.zeros((labels.size, labels.max() + 1))
labels_one_hot[np.arange(labels.size), labels] = 1

with open(data_file_path) as f:
    data_text = f.readlines()
data = np.array([np.fromstring(text, dtype=int, sep=',') for text in data_text])

data = np.array(data[permutation])

labels = labels[:2000]
counted_labels = [np.sum(labels == i) for i in range(10)]
print(counted_labels)
labels_one_hot = labels_one_hot.astype(np.float32)[:2000]
data = data.astype(np.float32)[:2000]

neural_network = Normal_model(**nn_structure)
for generation in range(100):
    batch_test_labels = labels[:1000]
    batch_test_data = data[:1000]
    predictions = neural_network.p_forward_pass(batch_test_data)
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == batch_test_labels)
    # params = neural_network.get_parameters()
    print(f"Generation {generation}, accuracy: {accuracy}")

    for batch_start in range(0, len(data) - batch_size, batch_size):
        batch_data = data[batch_start:batch_start + batch_size]
        batch_labels = labels_one_hot[batch_start:batch_start + batch_size]
        neural_network.backward_SGD(batch_data, batch_labels, lr, loss)


