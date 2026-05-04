from abc import ABC, abstractmethod
from typing import Any


class BaseService(ABC):
    def __init__(self, path: str, root_path: str, verbose: bool = False):
        """Generic read/write service for files
        Args:
            path (str): Path for the file.
            root_path (str): Root path for the file.
            verbose (bool, optional): should user information be displayed? Defaults to False
        """
        self.path = path
        self.root_path = root_path
        self.verbose = verbose
        # __metaclass__ = abc.ABCMeta

    @abstractmethod
    def doRead(self, **kwargs: Any) -> Any:
        """Abstract method to read data from a file"""
        raise NotImplementedError("Subclasses must implement doRead method")

    @abstractmethod
    def doWrite(self, X: Any, **kwargs: Any) -> Any:
        """Abstract method to write data to a file
        Args:
            X: input data
        """
        raise NotImplementedError("Subclasses must implement doWrite method")
