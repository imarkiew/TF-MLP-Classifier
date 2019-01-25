from Tools import prepare_data
from NeuralNetwork import learn_neural_network
from Tools import plot_loss
from NeuralNetwork import predict_output

file_path = "./data/diabetes.csv"
is_header_present = True
name_or_number_of_target_column = "class"
separator = ","
batch_size = 4
n = 4
hidden = [n]
biases = [0.0, 0.0] #biases vector should have size len(hidden) + 1 (last layer is before softmax and has a size num_of_classes)
is_oversampling_enabled = True
percent_of_test_examples = 0.3
polynomial_features_degree = None
learning_rate = 0.001
dropout_prob = 0.5
number_of_epochs = 200
type_of_f1_score = "macro"
is_model_saved = True
model_path = "./models/model"

# file_path = "./data/iris.data"
# is_header_present = False
# name_or_number_of_target_column = 5
# separator = ","
# batch_size = 4
# n = 2
# hidden = [n]
# biases = [0.0, 0.0] #biases vector should have size len(hidden) + 1 (last layer is before softmax and has a size num_of_classes)
# is_oversampling_enabled = True
# percent_of_test_examples = 0.3
# polynomial_features_degree = None
# learning_rate = 0.001
# dropout_prob = 0.5
# number_of_epochs = 200
# type_of_f1_score = "macro"
# is_model_saved = True
# model_path = "./models/model"

# file_path = "./data/LungCancer_25.csv"
# is_header_present = False
# name_or_number_of_target_column = 1
# separator = ";"
# batch_size = 4
# n = 4
# hidden = [n]
# biases = [0.0, 0.0] #biases vector should have size len(hidden) + 1 (last layer is before softmax and has a size num_of_classes)
# is_oversampling_enabled = False
# percent_of_test_examples = 0.3
# polynomial_features_degree = None
# learning_rate = 0.001
# dropout_prob = 0.5
# number_of_epochs = 200
# type_of_f1_score = "macro"
# is_model_saved = True
# model_path = "./models/model"

# file_path = "./data/reprocessed.hungarian.data"
# is_header_present = False
# name_or_number_of_target_column = 14
# separator = " "
# batch_size = 4
# n = 4
# hidden = [n]
# biases = [0.0, 0.0] #biases vector should have size len(hidden) + 1 (last layer is before softmax and has a size num_of_classes)
# is_oversampling_enabled = True
# percent_of_test_examples = 0.3
# polynomial_features_degree = None
# learning_rate = 0.001
# dropout_prob = 0.5
# number_of_epochs = 200
# type_of_f1_score = "macro"
# is_model_saved = True
# model_path = "./models/model"

Xx, Xt, yy, yt, enc = prepare_data(file_path, is_header_present, name_or_number_of_target_column,
                 separator, percent_of_test_examples, is_oversampling_enabled,
                 polynomial_features_degree)

losses = learn_neural_network(Xx, yy, Xt, yt, learning_rate, dropout_prob, biases, hidden, batch_size, number_of_epochs,
                              enc, type_of_f1_score, is_model_saved, model_path)
plot_loss(losses, True, "./models/losses.png")

# restore model and check predicted output for all examples in set
# x, y, y_one_hot, enc = prepare_data(file_path, is_header_present, name_or_number_of_target_column,
#                  separator, 0, False, polynomial_features_degree)
# path_to_model = "./models/model.meta"
# check_point_path = "./models/"
# print(predict_output(x, path_to_model, check_point_path, enc))
# print(y)