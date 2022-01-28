# Acoustic-Features
audio/speech feature extraction using parselmouth, librosa, disvoice


### praat.py

praat을 python에서 사용할 수 있게 구현한 parselmouth을 이용한 피쳐 추출.
slicing 방법 추가
- pitch
- f1, f2, f3
- intensity
- jitter
- shimmer
- hnr
- mfcc
- center of gravity
- standard deviation
- skewness
- kurtosis

### librosa.py
librosa에서만 얻을 수 있는 피쳐들을 뽑아냅니다. 
spectrogram, formant 등은 더 자세하게 쪼갤 수 있는 praat을 이용

- Centroid
- Bandwidth
- Contrast
- Flatness
- Rolloff
- Zero-Crossing
- RMS
- Envelope

### glottal.py

disvoice에서 제공하는 feature들.
waveform을 통해 성대의 충돌음과 같은 피쳐들을 추출함

- GCI
- NAQ
- QOQ
- H1H2
- HRF

