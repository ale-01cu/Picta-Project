from abc import ABC

class AbstractStage(ABC):
    model: object
    name: str

    def inputs(self):
        raise NotImplementedError()
    def outputs(self):
        raise NotImplementedError()