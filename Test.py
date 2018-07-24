from Tools import prepare_data
from NeuralNetwork import learn_neural_network
from Tools import plot_loss
from NeuralNetwork import predict_output

name_of_file = "./Data/diabetes.csv"
is_header_present = True
name_or_number_of_target_column = "class"
separator = ","
batch_size = 8
n = 4
hiddens = [n]
biases = [0.0, 0.0] #biases vector should have size len(hiddens) + 1 (last layer is predictive)
is_oversampling_enabled = True
is_polynomial_features_enabled = False

# name_of_file = "./Data/iris.data"
# is_header_present = False
# name_or_number_of_target_column = 5
# separator = ","
# batch_size = 8
# n = 4
# hiddens = [n]
# biases = [0.0, 0.0]
# is_oversampling_enabled = True
# is_polynomial_features_enabled = True

# name_of_file = "./Data/LungCancer_25.csv"
# is_header_present = False
# name_or_number_of_target_column = 1
# separator = ";"
# batch_size = 4
# n = 4
# hiddens = [n, n]
# biases = [0.0, 0.0, 0.0]
# is_oversampling_enabled = False #The feature is disabled because there are too few samples of the minority class
# is_polynomial_features_enabled = False

# name_of_file = "./Data/reprocessed.hungarian.data"
# is_header_present = False
# name_or_number_of_target_column = 14
# separator = " "
# batch_size = 8
# n = 4
# hiddens = [n, n]
# biases = [0.0, 0.0, 0.0]
# is_oversampling_enabled = True
# is_polynomial_features_enabled = True

percent_of_test_examples = 0.3
polynomial_features_degree = 2
learning_rate = 0.001
dropout_prob = 0.5
number_of_epochs = 300
type_of_f1_score = "macro"
is_model_saved = True
name_of_model = "model"

Xx, Xt, yy, yt, enc = prepare_data(name_of_file, is_header_present, name_or_number_of_target_column,
                 separator, percent_of_test_examples, is_oversampling_enabled, is_polynomial_features_enabled,
                 polynomial_features_degree)

losses = learn_neural_network(Xx, yy, Xt, yt, learning_rate, dropout_prob, biases, hiddens, batch_size, number_of_epochs,
                              enc, type_of_f1_score, is_model_saved, name_of_model)
plot_loss(losses, True, "losses.png")

#restore model and check predicted output for all examples in set
# x, y, y_one_hot, enc = prepare_data(name_of_file, is_header_present, name_or_number_of_target_column,
#                  separator, 0, False, is_polynomial_features_enabled,
#                  polynomial_features_degree)
# model_path = "model.meta"
# check_point_path = "./"
# print(predict_output(x, model_path, check_point_path, enc))
# print(y)

