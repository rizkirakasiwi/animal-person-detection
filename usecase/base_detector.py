from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        pass
