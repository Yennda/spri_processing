import tools as tl
import global_var

class NanoParticle():
    def __init__(self, np_id, first_frame, positions, positive):
        self.np_id = np_id
        self.positions = positions
        self.first_frame = first_frame

        self.color = tl.random_color()
        self.positive = positive
        self.last_frame = first_frame + len(positions)

    def position(self, f):
        # if f == self.first_frame + len(self.positions)//2:
        #     self.color = global_var.red
        # else:
        #     self.color = global_var.green
        return self.positions[f - self.first_frame][::-1]
