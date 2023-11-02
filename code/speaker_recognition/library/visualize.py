
import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(matrix, xaxis, yaxis, xunits='time (s)', yunits='Hz', zunits='(dB rel.)'):
    """plot_matrix(matrix, xaxis, yaxis, xunits, yunits
    Plot a matrix.  Label columns and rows with xaxis and yaxis respectively
    Intensity map is labeled zunits.
    Put "" in any units field to prevent plot
    """

    # Plot the matrix as a mesh, label axes and add a scale bar for
    # matrix values
    plt.pcolormesh(xaxis, yaxis, matrix)
    plt.xlabel(xunits)
    plt.ylabel(yunits)
    plt.colorbar(label=zunits)


def plot_spectrogram(specgram, adv_ms, len_ms, labels=None):
    """

    :param specgram:  Assumes spectrogram is frequency X frames
    :param adv_ms: frame advance in ms
    :param len_ms: frame length in ms
    :return:
    """

    print(specgram.shape)

    adv_s = adv_ms / 1000.0
    len_s = len_ms / 1000.0

    # Create the time axis in seconds
    t_N = specgram.shape[1]
    taxis = np.arange(0, t_N*adv_s, adv_s)

    # Spectrogram only goes up to the Nyquist rate.  Compute frequency
    # bin width and generate axis
    bin_Hz = 1 / len_s  # Divide sample length into 1 s for bin width
    freq_n = specgram.shape[0]
    faxis = [f * bin_Hz for f in range(freq_n)]

    plot_matrix(specgram, taxis, faxis)

    if labels is not None:
        # Add * at mean frequency where labels are True
        label_indices = np.where(labels == True)[0]
        mean_Hz = np.ones(np.shape(label_indices)) * np.mean(faxis)
        plt.plot(taxis[label_indices], mean_Hz, '*')

    plt.show(block=True)


def plot_training_history(history):

    # plot training & validation loss values
    plt.figure(figsize=(12, 4))

    # plot training & validation losses
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # plot training & validation accuracy values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'train_validation.png')
    plt.tight_layout()
    plt.show()
    # plt.show(block=True)
