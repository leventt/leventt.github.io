import os
import glob
import torch

from constants import ROOT
from constants import AUDIOSEQ_LEN
from constants import EXTRACTOR_LEN


if __name__ == "__main__":
    DEVICE = torch.device("cpu")

    MODEL_CP_PATH = os.path.join(ROOT, "model", "*", "*.pth")
    MODEL_CP_PATH = glob.glob(MODEL_CP_PATH)[-1]
    print(MODEL_CP_PATH)

    model = torch.load(MODEL_CP_PATH)
    model.eval()
    model.to(DEVICE)

    tracedScript = torch.jit.trace(
        model,
        (torch.zeros((270, 1, AUDIOSEQ_LEN, EXTRACTOR_LEN))),
        strict=False,
    )

    tracedScript.save(os.path.join(ROOT, "avaz.pt"))

    torch.onnx.export(
        model,
        (torch.zeros((270, 1, AUDIOSEQ_LEN, EXTRACTOR_LEN))),
        os.path.join(ROOT, "avaz.onnx"),
        opset_version=13,  # latest at the time of writing
    )
