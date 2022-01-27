"""
librosa에서만 얻을 수 있는 피쳐들을 뽑아냅니다. 
spectrogram, formant 등은 더 자세하게 쪼갤 수 있는 praat을 이용합니다. 
"""


#!pip install librosa

import librosa
import numpy as np
from scipy.signal import hilbert

def librosa_feats(self, wv):
    y, sr = librosa.load(wv)
    ## spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    # contrast
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)[0]
    # flatness
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    # rolloff
    # Approximate maximum frequencies with roll_percent=0.85 (default)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    # zero-crossing
    zc = librosa.feature.zero_crossing_rate(y)[0]
    # rms: energy
    rms = librosa.feature.rms(y=y)[0]
    # envelope
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    return {'Centroid':cent, 'Bandwidth':spec_bw, 'Contrast':contrast, 'Flatness':flatness, 
            'Rolloff':rolloff, 'Zero-Crossing':zc, 'RMS': rms, 'Envelope':amplitude_envelope}