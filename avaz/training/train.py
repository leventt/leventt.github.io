import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from data import Data
from data import getAudioFeatures
from model import Model
from model import getInference
from render import getValidationVideoTensor

from constants import ROOT
from constants import DEVICE
from constants import AUDIOSEQ_LEN
from constants import EXTRACTOR_LEN
from constants import FPS
from constants import SHAPES_LEN


def prep():
    dataSet = Data()
    model = Model().to(DEVICE)
    model.train()

    return model, dataSet


def loop(batchSize, learningRate, model, dataSet, scheduler=False, epochCount=1000000):
    # DON'T SHUFFLE SO MOTION LOSS CAN BE CALCUALTED
    dataLoader = DataLoader(dataset=dataSet, batch_size=batchSize, shuffle=False)

    optimizerSettings = [{"params": model.parameters(), "lr": learningRate}]
    optimizer = torch.optim.Adam(optimizerSettings)
    if scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.97, verbose=True
        )

    runStr = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    runStr = f"{runStr}_{batchSize}_{learningRate}"
    logWriter = SummaryWriter(os.path.join(ROOT, "logs", runStr))

    modelDir = os.path.join(ROOT, "model", runStr)
    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    validationAudioPath = os.path.join(ROOT, "data", "validation.wav")
    validationAudioFeatures = getAudioFeatures(validationAudioPath)[: FPS * 4]
    validationMatcap = Image.open(os.path.join(ROOT, "data", "matcap.png"))
    trainingAudioFeatures = dataSet.inputSpeechFeatures[: FPS * 4]

    print("start: {}".format(datetime.now()))
    start = datetime.now()
    print("training")

    criterion = torch.nn.HuberLoss(reduction="none").to(DEVICE)
    for epochIdx in range(epochCount):
        averageMotionLoss = 0.0
        averageShapeLoss = 0.0
        for inputData, targetData in dataLoader:
            inputData = inputData.view(-1, 1, AUDIOSEQ_LEN, EXTRACTOR_LEN).to(DEVICE)
            targetData = targetData.view(-1, SHAPES_LEN).to(DEVICE)

            optimizer.zero_grad()
            modelResult = model(inputData)

            shapeLoss = criterion(modelResult, targetData).sum(dim=-1).mean()
            motionLoss = (
                criterion(
                    torch.roll(modelResult, -1)[:-1] - modelResult[:-1],
                    torch.roll(targetData, -1)[:-1] - targetData[:-1],
                )
                .sum(dim=-1)
                .mean()
            )

            (shapeLoss + motionLoss).backward()
            optimizer.step()

            averageMotionLoss += motionLoss.item()
            averageShapeLoss += shapeLoss.item()

        averageMotionLoss /= len(dataLoader)
        averageShapeLoss /= len(dataLoader)

        logWriter.add_scalar("motion", averageMotionLoss, epochIdx + 1)
        logWriter.add_scalar("shape", averageShapeLoss, epochIdx + 1)

        if (epochIdx + 1) % 100 == 0:
            with torch.no_grad():
                torch.save(
                    model,
                    os.path.join(modelDir, "{}_E{:09d}.pth".format(runStr, epochIdx + 1)),
                )
                logWriter.add_video(
                    "validation",
                    getValidationVideoTensor(
                        getInference(model, validationAudioFeatures), validationMatcap,
                    ),
                    epochIdx + 1,
                    fps=30,
                )
                logWriter.add_video(
                    "training",
                    getValidationVideoTensor(
                        getInference(model, trainingAudioFeatures), validationMatcap,
                    ),
                    epochIdx + 1,
                    fps=30,
                )
        if scheduler:
            scheduler.step()

    print("done")
    print("duration: {}".format(datetime.now() - start))


if __name__ == "__main__":
    model, dataSet = prep()
    loop(4096, 1e-5, model, dataSet)
