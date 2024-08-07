from abc import ABC

class AbstractStage(ABC):
    def inputs(self):
        raise NotImplementedError()
    def outputs(self):
        raise NotImplementedError()