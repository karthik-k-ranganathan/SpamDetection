import numpy as np


class ModelInfo:
    def __init__(self) -> None:
        self.ModelName = "No Model Selected"
        self.Model = None
        self.Tokenizer = None
        self.AccuracyScore = np.nan
        self.ModelTrainedVer = "0.0.0.0"
