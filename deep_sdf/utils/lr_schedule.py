# Learning rate schedulers:
# https://github.com/facebookresearch/DeepSDF/blob/master/train_deep_sdf.py#L18

class LearningRateSchedule:
    def get_learning_rate(self, epoch: int) -> float:
        pass


class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value: float):
        self.value = value

    def get_learning_rate(self, epoch: int) -> float:
        return self.value


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial: float, interval: int, factor: float):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch: int) -> float:
        return self.initial * (self.factor ** (epoch // self.interval))