'''
**************************
CS682 - Assignment 2, Part 2
29.09.2023, San Diego
Code Version 1.0
**************************
I promise that the attached assignment is my own work.
I recognize that should this not be the case,
I will be subject to penalties as outlined in the course syllabus.
Thomas Keller
'''

from argparse import ArgumentParser

import numpy as np
# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib
# add-on packages


# project imports
from library.corpus import King
from library.audio_io import read_wav
from library.timer import Timer
from feature_extraction import get_features
from architecture import get_model
from classifier import train, test
from library.visualize import plot_training_history


def main(args):
    # Specify how matplotlib renders (backend library)
    # MacoOS:  'MacOSX' - native MacOS
    # TkAgg - Tcl/Tk backend
    # Qt5Agg:  Qt library, must be installed, e.g. module pyside2
    matplotlib.use("TkAgg")
    plt.ion()

    # select only Nutley
    king = King(args.king, group="Nutley")  # create instance of King corpus

    # get speakers of class king
    list_speakers = king.get_speakers()
    print(list_speakers)

    # list for features and labels training
    all_features_training = []
    all_labels_training = []

    # get features from speakers
    for speaker in list_speakers:
        # speaker path files
        speaker_files = king[speaker]
        print('Speaker: ' + str(speaker))
        print(speaker_files)

        # get half amount of files
        num_files_half = len(king.__getitem__(speaker))//2

        # get feature for first half of recordings for training
        for file in speaker_files[:num_files_half]:

            print('file: ' + str(file))

            # get features and labels of first half of recordings
            features, labels = get_features(file, args.adv_ms, args.len_ms, speaker, debug=False)

            # remap labels 0-N-1
            remapped_labels = [king.speaker_category(label) for label in labels]

            all_features_training.append(features)
            all_labels_training.append(remapped_labels)

    print('-----------------')
    # concatenate features and labels
    concatenated_features = np.concatenate(all_features_training, axis=0)
    concatenated_labels = np.concatenate(all_labels_training, axis=0)

    # create one hot labels
    one_hot_labels = to_categorical(concatenated_labels)

    # get model parameters
    features_n = concatenated_features.shape[1]  # number of features: 81
    output_n = one_hot_labels.shape[1]  # number of output classes: 25
    hidden_n = 3  # number of hidden layers
    width_n = 350  # number of neurons in each hidden layer
    l2_penalty = 0.0001
    learning_rate = 0.001

    # print Architecture
    print('Architecture:')
    print(features_n, hidden_n, width_n, l2_penalty, output_n)

    # filename with parameters
    file_string = f'f{features_n}_out{output_n}_hid{hidden_n}_w{width_n}_l2{l2_penalty}'

    # get model from architecture file
    model = get_model('l2', features_n, hidden_n, width_n, l2_penalty, output_n)

    # train model
    history = train(model, concatenated_features, one_hot_labels, learning_rate, epochs_n=10)

    # plot model training history - loss/accuracy
    plot_training_history(history)

    # test model, get error rate
    error_rate = test(model, list_speakers, file_string, king, args.adv_ms, args.len_ms,)
    print(f'Error Rate: {error_rate * 100:.2f}%')


if __name__ == "__main__":
    # print('start')
    # Process command-line arguments
    parser = ArgumentParser(
        prog="Speaker Identification",
        description="Classify speech to speaker")
    parser.add_argument("-k", "--king", required=True,
                        action="store",
                        help="King corpus directory")
    parser.add_argument("-l", "--len_ms", action="store",
                        type=float, default=20,
                        help="frame length in ms")
    parser.add_argument("-a", "--adv_ms", action="store",
                        type=float, default=10,
                        help="frame  advance in ms")

    args = parser.parse_args()

    main(args)
