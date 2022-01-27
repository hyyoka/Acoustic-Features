"""
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
"""

#!pip install textgrid praat-parselmouth

from parselmouth import Sound
from parselmouth.praat import call
import numpy as np

class FeatureGenerator:
    def __init__(self, sound): # sound = Sound(wav_file)
        self.sample_rate = 16_000
        self.fft_size = 512
        self.window_size = 400
        self.hop_size = 160
        self.time_step = self.hop_size/self.sample_rate
        self.call = parselmouth.praat.call
        self.high_resolution_desired = False
        

    def _get_wv_feats(self, sound):
        pre_emphasis = call(sound, 'Filter (pre-emphasis)', 80)
        spectrum = call(pre_emphasis, 'To Spectrum', 'yes')
        cog = call(spectrum, 'Get centre of gravity', 2) # Center of gravity'
        std = call(spectrum, 'Get standard deviation', 2)
        skw = call(spectrum, 'Get skewness', 2)
        kur = call(spectrum, 'Get kurtosis', 2)
        return {'KUR':kur,'SkW':skw,'COG':cog,'SD':std}

    def _get_mfcc(self, sound, start, end): 
        """mfcc 추출: pre-emphasis => window => DFT => MelFilterBank => log => IDFT """

        mfcc = sound.to_mfcc(number_of_coefficients=12, time_step=self.time_step, window_length=self.window_size/self.sample_rate, maximum_frequency=7600)
        mfcc_arr = mfcc.to_array()[1:] # 0 index is the energy of cepstrum
        mfcc_bins = mfcc.x_bins()[:, 0]

        def to_frame(time):
            frame = np.searchsorted(mfcc_bins, time)-1
            return frame if frame >= 0 else 0

        start = to_frame(start)
        end = to_frame(end)

        _mfcc = mfcc_arr[:, start:end+1]
        _mfcc = np.mean(_mfcc, axis=-1)
        return _mfcc

    def _get_formant(self, sound, start, end, get_part = True):
        """
        formant추출
        get_part: 가운데 3분의 1 추출
        """
        formant = sound.to_formant_burg(time_step=self.time_step, max_number_of_formants=4.5, maximum_formant=4700.0, window_length=self.window_size/self.sample_rate, pre_emphasis_from=50)
        duration = end - start
        _formant = {}
        for f in range(3):  # f1~f3
            formant_ls = []
            for i in range(3,100,3): # 총 33개의 구간 
                formant_ls.append(formant.get_value_at_time(formant_number=f+1, time=start + i/100*duration))
            if get_part:
                formant_ls = formant_ls[10:-11] 
            formant_name = "f"+str(f+1)
            _formant[formant_name] = formant_ls
        return _formant

        
    def _get_pitch(self, sound, start, end, get_part = True):
        """f0 추출"""

        duration = end - start     
        pitch = sound.to_pitch(time_step=self.time_step, pitch_floor=75.0, pitch_ceiling=600.0)

        _ff = []
        for i in range(3,100,3): # 총 24개의 구간: 8-8-8
          _ff.append(pitch.get_value_at_time(time=start + i/100*duration))
        if get_part:
          _ff = _ff[10:-11]

        return _ff

    def _get_intensity(self, sound, start, end, get_part = True):
        """intensity 추출"""

        duration = end - start
        intensity = sound.to_intensity(minimum_pitch=100.0, time_step=self.time_step, subtract_mean=True)
        _int = []
        for i in range(3,100,3):
            _int.append(intensity.get_value(time=start + i/100*duration))
        if get_part:
          _int = _int[10:-11]

        return _int

    def _get_jitter(self, sound, start, end):
        """jitter 추출"""
        point_process = self.call(sound, "To PointProcess (periodic, cc)", 75, 600)  # pitch_floor=75, pitch_ceiling=600
        local_jitter = self.call(point_process, "Get jitter (local)", start, end, 0.0001, 0.02, 1.3)
        localabsolute_jitter = self.call(point_process, "Get jitter (local, absolute)", start, end, 0.0001, 0.02, 1.3)
        rap_jitter = self.call(point_process, "Get jitter (rap)", start, end, 0.0001, 0.02, 1.3)
        ppq5_jitter = self.call(point_process, "Get jitter (ppq5)", start, end, 0.0001, 0.02, 1.3)
        ddp_jitter = self.call(point_process, "Get jitter (ddp)", start, end, 0.0001, 0.02, 1.3)

        _jitter = [local_jitter, localabsolute_jitter, rap_jitter, ppq5_jitter, ddp_jitter]

        return _jitter

    def _get_shimmer(self, sound, start, end):
        """shimmer 추출"""
        point_process = self.call(sound, "To PointProcess (periodic, cc)", 75, 600)  # pitch_floor=75, pitch_ceiling=600
        local_shimmer = self.call([sound, point_process], "Get shimmer (local)", start, end, 0.0001, 0.02, 1.3, 1.6)
        localdb_shimmer = self.call([sound, point_process], "Get shimmer (local_dB)", start, end, 0.0001, 0.02, 1.3, 1.6)
        apq3_shimmer = self.call([sound, point_process], "Get shimmer (apq3)", start, end, 0.0001, 0.02, 1.3, 1.6)
        apq5_shimmer = self.call([sound, point_process], "Get shimmer (apq5)", start, end, 0.0001, 0.02, 1.3, 1.6)
        apq11_shimmer = self.call([sound, point_process], "Get shimmer (apq11)", start, end, 0.0001, 0.02, 1.3, 1.6)
        dda_shimmer = self.call([sound, point_process], "Get shimmer (dda)", start, end, 0.0001, 0.02, 1.3, 1.6)
        _shimmer = [local_shimmer, localdb_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer]

        return _shimmer

    def _get_hnr(self, sound, start, end, method='cc', get_part=True):
        """
        Calculate Harmonics-to-Noise Ratio (HNR); represents the degree of acoustic periodicity and Voice quality
        """
        if method == 'ac': 
            hnr= sound.to_harmonicity_ac(time_step=self.time_step) # cross-correlation method (preferred).
        else: 
            hnr= sound.to_harmonicity_cc(time_step=self.time_step) # cross-correlation method (preferred).

        duration = end - start
        _hnr= []

        for i in range(3,100,3):
            _hnr.append(hnr.get_value(time=start + i/100*duration))
        if get_part:
          _hnr = _hnr[10:-11]
        _hnr = [h for h in _hnr if h != -200]
        return _hnr
    