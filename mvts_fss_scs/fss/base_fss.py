import os
from abc import ABC, abstractmethod

import numpy as np
from CONSTANTS import DATA_PATH


class BaseFSS(ABC):

    def __init__(self, *args):
        self.data = np.load(os.path.join(DATA_PATH, "partition1.npz"))

    @abstractmethod
    def rank(self):
        pass
