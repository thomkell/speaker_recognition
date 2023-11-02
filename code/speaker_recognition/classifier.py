

# add-on modules

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_extraction import get_features

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.optimizers import Adam


def train(model, features, labels, lr, epochs_n=20):

    # split  dataset: training/validation
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.10, random_state=42)

    # define optimizer
    optimizer = Adam(learning_rate=lr)

    # compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs_n) # batch_size=100

    return history


# predict speaker by file
def test(model, list_speakers, filename, king, adv_ms, len_ms):

    # correct predictions and total predictions
    correct_predictions = 0
    total_predictions = 0

    # true speakers and predicted speaker
    all_true_speakers = []
    all_predicted_speakers = []

    # get features from speakers
    for speaker in list_speakers:
        speaker_files = king[speaker]

        # half of speaker
        num_files_half = len(king.__getitem__(speaker)) // 2

        # get features for the second half of recordings for testing
        for file in speaker_files[num_files_half:]:

            # get features from feature extraction
            features, labels = get_features(file, adv_ms, len_ms, speaker, debug=False)

            # predict probabilities for the current file
            predicted_probabilities = model.predict(features)
            # change to log matrix form
            log_probs = np.log(predicted_probabilities + 1e-6)
            # sum columns, axis=0
            summed_log_probs = np.sum(log_probs, axis=0)
            # max value
            predicted_speaker = np.argmax(summed_log_probs)
            print(predicted_speaker)

            # remap labels 0-N-1
            true_speaker = king.speaker_category(labels[0])

            # save true speaker and predicted speaker lists
            all_true_speakers.append(true_speaker)
            all_predicted_speakers.append(predicted_speaker)

            # check if speaker is right predicted
            if true_speaker == predicted_speaker:
                correct_predictions += 1
            total_predictions += 1

            # print the prediction for the current file
            print(f"True Speaker: {true_speaker}, Predicted Speaker: {predicted_speaker}")

    # Calculate error rate
    error_rate = 1 - (correct_predictions / total_predictions)

    # remap labels for confusion matrix
    remapped_labels = [king.speaker_category(label) for label in list_speakers]

    # Ensure remapped_labels is sorted
    remapped_labels = sorted(remapped_labels)

    # Calculate and display the confusion matrix
    confusion = confusion_matrix(all_true_speakers, all_predicted_speakers)
    conf_disp = ConfusionMatrixDisplay(confusion, display_labels=remapped_labels)

    conf_disp.plot(cmap='Blues')

    plt.title(f'Confusion matrix\n architecture: {filename}\n ER: {error_rate * 100:.2f}%')

    # Rotate x-axis labels by 90 degrees
    plt.xticks(rotation=90)

    # plt.title(f'Confusion matrix, architecture: {filename}, ER: {error_rate * 100:.2f}%')
    plt.savefig(f'confMatrix/confusionMatrix_{filename}_{error_rate * 100:.0f}.png')
    plt.show(block=True)

    return error_rate

