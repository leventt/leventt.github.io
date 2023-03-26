import torch
from torch import nn

from constants import FORMANT_LEN
from constants import SHAPES_LEN


def getInference(model, *args, **kwargs):
    with torch.no_grad():
        result = model(*args, **kwargs).view(-1, SHAPES_LEN)
    return torch.clamp(result, 0, 1).detach().cpu().numpy()


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.formantAnalysis = nn.Sequential(
            nn.Conv2d(1, 7, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Dropout2d(0.1),
            nn.Conv2d(7, 13, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(13),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Dropout2d(0.1),
            nn.Conv2d(13, 23, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(23),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Dropout2d(0.1),
            nn.Conv2d(23, 27, (1, 3), (1, 2), (0, 1), 1),
            nn.BatchNorm2d(27),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(27, FORMANT_LEN, (1, 2), (1, 2)),
            nn.BatchNorm2d(FORMANT_LEN),
        )
        self.articulation = nn.Sequential(
            nn.Conv2d(FORMANT_LEN, FORMANT_LEN, (3, 1), (2, 1), (1, 0), 1,),
            nn.LeakyReLU(),
            nn.Conv2d(FORMANT_LEN, FORMANT_LEN, (3, 1), (2, 1), (1, 0), 1,),
            nn.LeakyReLU(),
            nn.Conv2d(FORMANT_LEN, FORMANT_LEN, (3, 1), (2, 1), (1, 0), 1,),
            nn.LeakyReLU(),
            nn.Conv2d(FORMANT_LEN, FORMANT_LEN, (3, 1), (2, 1), (1, 0), 1,),
            nn.LeakyReLU(),
            nn.Conv2d(FORMANT_LEN, FORMANT_LEN, (3, 1), (2, 1), (1, 0), 1,),
            nn.LeakyReLU(),
            nn.Conv2d(FORMANT_LEN, FORMANT_LEN, (4, 1), (4, 1), (1, 0), 1,),
        )
        self.output = nn.Sequential(
            nn.Linear(FORMANT_LEN, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, SHAPES_LEN),
            nn.Sigmoid(),
        )

    def forward(self, inp):
        # inp.shape = [BATCH_SIZE, 1, AUDIOSEQ_LEN, EXTRACTOR_LEN]
        out = self.formantAnalysis(inp)
        # out.shape = [BATCH_SIZE, FORMANT_LEN, AUDIO_SEQLEN, 1]
        out = self.articulation(out)
        # out.shape = [BATCH_SIZE, FORMANT_LEN, 1, 1]
        out = self.output(out.view(-1, FORMANT_LEN))
        # out.shape = [BATCH_SIZE, SHAPES_LEN]
        return out.view(-1, SHAPES_LEN)
