import librosa
import numpy as np


def get_features(filename):
    '''
    Returns stacked beat-synchronous features for a given file. 
    Features calculated are timbre[12], chroma[12] and max_loudness[1].
    '''
    y, sr = librosa.load(filename, sr=None)

    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # Calculate chroma
    C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # Make a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.amplitude_to_db(S, ref=np.max)

    # Next, we'll extract the top 12 Mel-frequency cepstral coefficients (MFCCs)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=12)

    # Now, let's run the beat tracker.
    tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

    # feature.sync will summarize each beat event by the mean feature vector within that beat
    timbre = librosa.util.sync(mfcc, beats)
    timbre = (timbre + 1000) / 1200  # ~ 0-1 normalization

    chroma = librosa.util.sync(C, beats, aggregate=np.median)
    max_loudness = librosa.util.sync(y, beats, aggregate=np.max)
    features = np.vstack([timbre, chroma, max_loudness])
    features = np.transpose(features)

    return features
