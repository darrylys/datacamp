
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs):
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

def plot_loss(train, validation, blurb='loss'):
    fig = plt.figure()
    plt.plot(train)
    plt.plot(validation)
    plt.title(f'model {blurb}')
    plt.ylabel(blurb)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"plot_{blurb}.png")
    plt.close(fig)

def plot_multiple(name_array_list, title):
    fig = plt.figure(figsize=(20, 6))
    for name, data, linestyle in name_array_list:
        plt.plot(data, linestyle=linestyle)
    plt.title(f'model {title}')
    plt.ylabel(title)
    plt.xlabel('epoch')
    plt.legend([tup[0] for tup in name_array_list], loc="upper left")
    plt.savefig(f"plot_{title}.png")
    plt.close(fig)

def create_tensorboard_callback(folder_name):
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + folder_name
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

