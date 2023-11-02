from sklearn.mixture import GaussianMixture
import numpy as np

# Add on modules (conda/pip)
import librosa.feature

# project modules
from .conversions import ms_to_samples


class GMM:
    """
    Gaussian mixture model
    Unsupervised learner
    """

    def __init__(self, data: np.array, mixtures: int,
                 covariance_type: str = 'diag',
                 max_iter: int = 50):
        """
        Estimate the GMM parameters
        :param data: N x Dim numpy array
        :param mixtures:  Number of mixtures
        :param covariance_type:  Restrictions on covariance estimation
           Common choices full, diag, tied, see sklearn.mixture.GaussianMixture
           for details
        :param max_iter:  Maximum number of expectation-maximization steps
        :return: None
        """

        if len(data.shape) == 1:
            # Reshape to explicitly make 2D
            data = data.reshape(-1, 1)
        self.model = GaussianMixture(
            mixtures, covariance_type=covariance_type, max_iter=max_iter)
        self.model.fit(data)

        return

    def predict_mixtureprob(self, data: np.array):
        """
        Predict probability of each data item given each mixture
        :param data: N x Dim numpy array
        :return: N x mixture array of probabilities
            Probability [r,m] is the likelihood of mixture m for row r
        """
        if len(data.shape) == 1:
            # Reshape to explicitly make 2D
            data = data.reshape(-1, 1)
        return self.model.predict_proba(data)

    def predict(self, data: np.array):
        """
        Predict class of each data item (highest likelihood mixture index)
        :param data: N x Dim numpy array
        :return: N x 1
            class [r] is the most likely mixture m for row r
        """
        if len(data.shape) == 1:
            # Reshape to explicitly make 2D
            data = data.reshape(-1, 1)
        return self.model.predict(data)

    def get_means(self):
        """
        Return the mean of each mixture
        :return:  mixture X Dim array of means, the m'th row is the
           mean of mixture m
        """
        return self.model.means_


def speech_detector(samples, Fs, adv_ms, len_ms):
    """
    Return a vector of indicator functions where [i]=True indicates that
    speech energy is predicted in the i'th frame contains speech
    :param samples:  pulse-code modulated samples from microphone
    :param Fs:  sample rate
    :param adv_ms:  frame advance (ms)
    :param len_ms:  frame length (ms)
    :return:  vector with one boolean entry per frame
    """

    # Convert to samples
    adv_n = ms_to_samples(Fs, adv_ms)
    len_n = ms_to_samples(Fs, len_ms)

    eps = 1e-5  # Small offset to prevent log 0
    # The frame advance is sometimes called the hop.
    # samples is N x channels, librosa expects channels by samples (transpose)
    rms_dB = 20 * np.log10(librosa.feature.rms(
        y=samples.T, hop_length=adv_n, frame_length=len_n) + eps)
    # librosa returns freqbins x channels x frames as it also handles
    # spectrogram energy.  Remove extraneous bin and transpose as GMM expects
    # the data the other way around... :-(
    rms_dB = np.squeeze(rms_dB, axis=0).T
    # Train a GMM to distinguish between two classes, speech and noise
    classes_n = 2
    endpointer = GMM(rms_dB, mixtures=classes_n)
    means = endpointer.get_means()
    # Assume mixture with higher intensity represents speech
    speech_idx = np.argmax(means)
    category = endpointer.predict(rms_dB)

    # Generate vector with True in frames corresponding to the speech mixture,
    # and False everywhere else
    # If you need noise labels, they can be derived with np.logical_not(speechI)
    speechI = category == speech_idx
    return speechI
