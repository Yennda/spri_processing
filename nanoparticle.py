import tools as tl
import numpy as np


class NanoParticle():
    def __init__(self, np_id, first_frame, positions):
        self.positions = positions
        self.first_frame = first_frame
        self.color = tl.random_color()

    def position(self, f):
        return self.positions[f - self.first_frame][::-1]
