import os
import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from scipy.ndimage import zoom as resample_ndsignal

from constants import SAMPLE_RATE
from constants import WINDOW_SIZE
from constants import WINDOW_STRIDE
from constants import AUDIOSEQ_LEN
from constants import EXTRACTOR_LEN
from constants import ROOT
from constants import DEVICE
from constants import FPS


def getAudioFeatures(path, expectedFrameCount=None):
    waveform, sampleRate = torchaudio.load(path)
    if sampleRate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sampleRate, SAMPLE_RATE)(waveform)
        sampleRate = SAMPLE_RATE  # just in case, this may be used later

    if expectedFrameCount is not None:
        waveform = torch.from_numpy(
            resample_ndsignal(
                waveform,
                zoom=(
                    1.0,
                    (expectedFrameCount / FPS * SAMPLE_RATE) / waveform.shape[1],
                ),
                order=1,
            )
        )

    if len(waveform.shape) > 1:
        waveform = waveform[0].to(DEVICE)
    waveform = torch.clamp(waveform, -1.0, 1.0)

    paddedWaveform = (
        torch.zeros(waveform.shape[0] + WINDOW_SIZE, dtype=waveform.dtype).to(DEVICE)
        + 0.5
    )
    paddedWaveform[WINDOW_SIZE // 2 : WINDOW_SIZE // 2 + waveform.shape[0]] = waveform
    features = torchaudio.compliance.kaldi.mfcc(
        paddedWaveform.unsqueeze(0),
        channel=-1,
        sample_frequency=SAMPLE_RATE,
        remove_dc_offset=True,
        window_type="hanning",
        num_ceps=EXTRACTOR_LEN,
        num_mel_bins=EXTRACTOR_LEN + 1,
        frame_length=WINDOW_SIZE,
        frame_shift=WINDOW_STRIDE,
    ).to(DEVICE)
    features -= features.min()
    features /= features.max()

    paddingLen = AUDIOSEQ_LEN
    if expectedFrameCount is not None:
        paddingLen += expectedFrameCount - features.shape[0]
    paddedFeatures = (
        torch.zeros(
            (features.shape[0] + paddingLen, EXTRACTOR_LEN), dtype=features.dtype
        ).to(DEVICE)
        + 0.5
    )
    paddedFeatures[paddingLen // 2 : paddingLen // 2 + features.shape[0]] = features
    audioFeatureSequences = (
        paddedFeatures.unfold(0, AUDIOSEQ_LEN, 1)
        .permute(0, 2, 1)
        .unsqueeze(1)
        .to(DEVICE)
    )
    if expectedFrameCount is not None:
        audioFeatureSequences = audioFeatureSequences[:expectedFrameCount]
    return audioFeatureSequences


class Data(Dataset):
    def __init__(self):
        inputSpeechPath = os.path.join(ROOT, "data", "data.wav")
        self.inputSpeechFeatures = getAudioFeatures(inputSpeechPath).to(DEVICE)

        self.targetShapes = (
            torch.from_numpy(np.load(os.path.join(ROOT, "data", "data.npy")))
            .float()
            .to(DEVICE)
        )

        self.count = min(self.targetShapes.shape[0], self.inputSpeechFeatures.shape[0])
        self.inputSpeechFeatures = (
            self.inputSpeechFeatures[: self.count].detach().to(DEVICE)
        )
        self.targetShapes = self.targetShapes[: self.count].detach().to(DEVICE)

    def __getitem__(self, i):
        if i < 0:  # for negative indexing
            i = self.count + i

        return (
            self.inputSpeechFeatures[i],
            self.targetShapes[i],
        )

    def __len__(self):
        return self.count
