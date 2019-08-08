"""
   Author: Javier Arellano-Verdejo
   Email: javier_arellano_verdejo@hotmail.com
   Date: 08/08/2019
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split

from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, GlobalAveragePooling1D, LSTM, Dropout

from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras import regularizers

from keras.optimizers import Adam
from keras import regularizers

number_of_classes = 2
input_dim = 0

def loadAndShuffleFile(filename, columns, p):
    """Load and shuffle the data."""

    # cargo el archivo de datos
    data = pd.read_csv(filename)[columns]

    # selecciono los datos de train y testing y los mezclo
    train, test = train_test_split(data, test_size=1-p, random_state=1)

    # devuelvo los datos
    return [train, test]


def normData(train, test):
    """Normalize data with mean 0 and standar deviation 1."""

    # descompongo el conjunto de datos en data y target
    d_train = train.iloc[:, :-1] # data
    t_train = train.iloc[:, -1]  # target

    d_test = test.iloc[:, :-1] # data
    t_test = test.iloc[:, -1]  # target

    # obtengo la media y desviación estandar del conjunto de datos de train
    mean = np.mean(d_train, axis=0)
    std = np.std(d_train, axis=0)

    # normalizo los conjuntos de train y test
    train_norm = (d_train - mean) / std
    test_norm =  (d_test - mean) / std

    return [train_norm, t_train], [test_norm, t_test]


def loadData(p=0.8):
    """Load data for training and testing."""

    # nombre de las columnas del set de datos que seran cargadas
    columns = ['rhos_412', 'rhos_469', 'rhos_555', 'rhos_645', 'rhos_859', 'rhos_1240', 'rhos_2130',
               'rhot_412', 'rhot_469', 'rhot_555', 'rhot_645', 'rhot_859', 'rhot_1240', 'rhot_2130',
               'class']

    # Paso 1: cargo los datos
    train, test = loadAndShuffleFile("./data/dataset.csv", columns, p)

    # Paso 3: mezclo los conjuntos de datos finales
    train = sklearn.utils.shuffle(train)
    test = sklearn.utils.shuffle(test)

    # Paso 4: normalizo los datos
    [train_data, train_target], [test_data, test_target] = normData(train, test)

    # Paso 5: convierto los DataFrames a arreglos
    train_data = np.array(train_data)
    train_target = np.array(train_target)
    test_data = np.array(test_data)
    test_test_target = np.array(test_target)

    return [train_data, train_target], [test_data, test_target]


def createModel_ERISNet():
    """Create the ERISNet sequencial model."""

    model = Sequential()
    model.add(Reshape(input_shape=(input_dim, 1), target_shape=(1, input_dim)))
    for i in range(3):
        model.add(Conv1D(64, 6, padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

    for i in range(3):
        model.add(Conv1D(128, 5, padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

    for i in range(3):
        model.add(Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))

    for i in range(2):
        model.add(LSTM(units=64, return_sequences=True, activation='tanh'))
        model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()

    plot_model(model, to_file='erisnet.png')

    return model

def mse(loss, val_loss):
    """Compute the Mean Square Error."""
    return ((loss - val_loss)**2).mean()

def main(filename):
    """Define main function."""
    
    global input_dim # contiene el numero de entradas de la red neuronal

    # loads and normalize the dataset
    [trainD, trainT], [testD, testT] = loadData(0.8)
    input_dim = trainD.shape[1]

    # Prepare dataset for training process
    trainD = trainD.reshape(trainD.shape[0], input_dim, 1)
    trainT = pd.get_dummies(trainT)

    # Prepare dataset for testing process
    testD = testD.reshape(testD.shape[0], input_dim ,1)
    testT = pd.get_dummies(testT)

    model = createModel_ERISNet()
    history = model.fit(trainD,
                        trainT,
                        batch_size=100,
                        epochs=100,
                        validation_data=(testD, testT))
    # Evaluates the trained neural network
    metrics = model.evaluate(testD, testT, verbose=0)

    # save model
    model.save(filename)

    # Shows the statistics and the training process
    print("Metrics Test Accuracy): %d%% " % (metrics[1]*100))

    # Gets historicsl information about the training process
    loss = history.history['acc']
    val_loss = history.history['val_acc']

    # Logs the training information
    np.savetxt("acc.csv", loss, delimiter=",")
    np.savetxt("val_acc.csv", val_loss, delimiter=",")

    # Shows information
    print("Max: %f " % (max(val_loss) * 100))
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b.', label='Training acc')
    plt.plot(epochs, val_loss, 'r.', label='Validation acc')

    # Fits polinomial model to the data-points
    z = np.polyfit(epochs, loss, 9)
    p1 = np.poly1d(z)
    plt.plot(epochs, p1(epochs), 'k-', linewidth=3, label='loss')

    z = np.polyfit(epochs, val_loss, 9)
    p2 = np.poly1d(z)
    plt.plot(epochs, p2(epochs), 'g-', linewidth=3, label='val_loss')

    # Computes the error between the optimization and generalization
    error = mse(p1(epochs), p2(epochs))
    print("MSE: %f " % error)

    # Plots results
    plt.title('Training and validation loss MSE:' + str(error))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("ERISNet V2")
    #main('erisnet_net.h5')
    createModel_ERISNet()