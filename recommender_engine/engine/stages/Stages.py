from abc import ABC

class Stages(ABC):
    def inputs(self):
        raise NotImplementedError()
    def outputs(self):
        raise NotImplementedError()