from emg2pose.data import Emg2PoseSessionData

class EmgStreamer:
    def __init__(self, data: Emg2PoseSessionData):
        self.emg = data['emg']
        self.t = 0
        self.buffer = []

    def step(self):
        sample = self.emg[self.t]
        self.t += 1

        self.buffer.append(sample)
        return sample