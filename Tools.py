import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt

def prepare_data(name_of_file, is_header_present, name_or_number_of_target_column, separator, percent_of_test_examples,
                 is_oversampling_enabled, polynomial_features_degree):
    if is_header_present:
        df = pd.read_csv(name_of_file, sep=separator)
        y = df[name_or_number_of_target_column].values
        df = df.drop(name_or_number_of_target_column, axis=1)
    else:
        df = pd.read_csv(name_of_file, header=None, sep=separator)
        y_classification = df.columns[int(name_or_number_of_target_column) - 1]
        y = df[y_classification].values
        df = df.drop(y_classification, axis=1)
    df = df.fillna(value=df.mean())
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    if polynomial_features_degree is not None:
        polynomial = PolynomialFeatures(polynomial_features_degree)
        X = polynomial.fit_transform(X)
    num_of_labels = len(find_outpu(y))
    enc = LabelEncoder()
    if percent_of_test_examples != 0:
        Xx, Xt, yy, yt = train_test_split(X, y, test_size=percent_of_test_examples, stratify=y)
        if is_oversampling_enabled:
            smt = SMOTE()
            Xx, yy = smt.fit_sample(Xx, yy)
            Xx, yy = shuffle(Xx, yy)
        yy_num = enc.fit_transform(yy)
        yt_num = enc.fit_transform(yt)
        yy_one_hot = np.eye(num_of_labels)[yy_num] #yy_num has the indexes of the binary vectors from np.eye(num_of_labels)
        yt_one_hot = np.eye(num_of_labels)[yt_num]
        return Xx, Xt, yy_one_hot, yt_one_hot, enc
    else:
        y_num = enc.fit_transform(y)
        y_one_hot = np.eye(num_of_labels)[y_num]
        return X, y, y_one_hot, enc

def find_outpu(y):
    return list(set(y))

def find_labels(y):
    for i in range(len(y)):
        index = np.argmax(y[i])
        y[i] = [0]*(len(y[i]))
        y[i][index] = 1
    return y

def decode_one_hot(y):
    return [i for element in y for i, yy in enumerate(element) if yy == 1]

def plot_loss(losses, is_plot_saved, path):
    epochs = range(1, len(losses[0]) + 1)
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(epochs, losses[0], "b-", label="Average loss on training set")
    ax.plot(epochs, losses[1], "r-", label="Average loss on validation set")
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('average loss', fontsize=15)
    ax.legend()
    if is_plot_saved:
        fig.savefig(path, bbox_inches='tight')
    plt.show()