# !git clone https://github.com/jcvasquezc/disvoice
# %cd disvoice
# !bash install.sh
"""
disvoice에서 제공하는 feature들.
waveform을 통해 성대의 충돌음과 같은 피쳐들을 추출함
"""

import pandas as pd
from disvoice.glottal.glottal import Glottal

glottalf=Glottal()

def _extract_glottal_f(wav_path):
    """문장 단위 피쳐 추출"""
    # metrics
    sent_feat=pd.DataFrame(glottalf.extract_features_file(wav_path, static=True, plots=False, fmt="csv"))
    metrics = ['global avg var GCI', 'global avg std NAQ','global avg std QOQ','global avg std H1H2',  'global avg std HRF']
    sent_features = sent_feat[metrics]
    sent_features.columns = ['GCI','NAQ','QOQ','H1H2','HRF']
    return sent_features