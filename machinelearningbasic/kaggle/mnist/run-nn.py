import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import runlib

def main2():
    df = pd.DataFrame(data={
        "label": [1, 1, 0, 0],
        "pixel1": [1, 1, 2, 2],
        "pixel2": [1, 1, 2, 2],
        "pixel3": [1, 1, 2, 2],
        "pixel4": [1, 1, 2, 2],
        "pixel5": [1, 1, 2, 2],
        "pixel6": [1, 1, 2, 2],
        "pixel7": [1, 1, 2, 2],
        "pixel8": [1, 1, 2, 2],
        "pixel9": [1, 1, 2, 2],
        "pixel10": [1, 1, 2, 2],
        "pixel11": [1, 1, 2, 2],
        "pixel12": [1, 1, 2, 2],
        "pixel13": [1, 1, 2, 2],
        "pixel14": [1, 1, 2, 2],
        "pixel15": [1, 1, 2, 2],
        "pixel16": [1, 1, 2, 2]
    })

    x = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    #x = x.values.reshape(x.shape[0], 4, 4, 1)
    x = np.reshape(x.values, (x.shape[0], 4, 4))
    print(x)
def main0():
    df = pd.DataFrame(data={
        "lalala": [100,200,300,400,500],
        "lilili": [10,20,30,40,50],
        "lululu": [1,2,3,4,5],
        "lelele": [0,1,0,1,0]
    })
    df = df / 100
    print(df)
    pass

def verysmol_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'verysmall', '-'

def smol_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'small', '--' 

def base_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'base', '-.'

def base_conv3_40_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Conv2D(40, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'base_conv340', ':'

def dropout_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'dropout', '-'

def dropout_conv3_40_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(40, (3,3), activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'dropout_conv340', '-.'

def starting_model(num_classes):
    return tf.keras.models.Sequential(layers=[
        tf.keras.layers.Conv2D(10, (3,3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(20, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(40, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(80, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ]), 'starting', '--'

def train_model(df):
    X = df.iloc[:,1:]
    y = df.iloc[:,0]

    # normalizing input from [0,255] to [0,1]
    X = X / 255

    # each row is image of 28x28, one channel.
    X = np.reshape(X.values, (X.shape[0], 28, 28, 1))
    y = y.values

    num_classes = 10
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    #models = [base_model(num_classes), verysmol_model(num_classes), 
    #    smol_model(num_classes), base_conv3_40_model(num_classes), 
    #    dropout_model(num_classes), dropout_conv3_40_model(num_classes)
    #]
    #models = [base_model(num_classes), verysmol_model(num_classes)]
    models = [starting_model(num_classes)]

    losses = []
    accuracies = []

    for model, model_name, line_style in models:
        model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

        #print(model.summary())

        #callbacks=[runlib.LossHistory(), runlib.AccuracyHistory(), runlib.create_tensorboard_callback('lr0.01')]
        callbacks=[runlib.create_tensorboard_callback('lr0.001_4lyrbn')]
        model.fit(X, y, batch_size=64, epochs=10, validation_split=0.2, callbacks=callbacks)

        #losses.append((f"train_{model_name}", callbacks[0].losses, line_style))
        #losses.append((f"val_{model_name}", callbacks[0].val_losses, line_style))
        #accuracies.append((f"train_{model_name}", callbacks[1].acc, line_style))
        #accuracies.append((f"val_{model_name}", callbacks[1].val_acc, line_style))

        #runlib.plot_loss(callbacks[0].losses, callbacks[0].val_losses, f'loss_{model_name}')
        #runlib.plot_loss(callbacks[1].acc, callbacks[1].val_acc, f'accuracy_{model_name}')

    runlib.plot_multiple(losses, 'loss')
    runlib.plot_multiple(accuracies, 'accuracy')

    return model


def main():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    #print(df_train.head())
    #print(df_test.head())

    #print(df_train['label'].value_counts())
    #x_train = df_train.drop('label', axis=1)

    #plt.figure(figsize=(12,10))
    #x, y = 10, 4
    #for i in range(40):  
    #    plt.subplot(y, x, i+1)
    #    plt.imshow(x_train.iloc[i, :].values.reshape((28,28)),interpolation='nearest')
    #plt.show()

    train_model(df_train)

    pass

if __name__ == '__main__':
    main()

