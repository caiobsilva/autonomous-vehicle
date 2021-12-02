import sys
from io import StringIO
from gym import utils
from gym_parking.envs import discrete
import numpy as np
from contextlib import closing

MAP = [
  "+-------------------+",
  "| |C|C|C| |C| | |C| |",
  "| |P| | | | | |P| |P|",
  "| |C|C| |C|C| | | |C|",
  "| | | | | | |C|C| |C|",
  "|C|P|P|C| |P| | | | |",
  "| | | |C| | |C|C|C| |",
  "| |C|C|C|P| |C|C|P| |",
  "| | | |P|C| |P| | | |",
  "|C|P| | | | | | | |C|",
  "|P|C| |C|C|x| |C| |P|",
  "+-------------------+",
]

class ParkingLotEnv(discrete.DiscreteEnv):
  metadata = {"render.modes": ["human", "ansi"]}

  def __init__(self):
    self.desc = np.asarray(MAP, dtype=b"c")
    states = 100
    rows = 10
    columns = 10
    row_limit = rows - 1
    col_limit = columns - 1
    states_arr = np.zeros(states)
    actions = 4

    P = { state: {action: [] for action in range(actions)} for state in range(states) }

    for row in range(rows):
      for col in range(columns):
        state = self.encode(row, col)

        if self.desc[1 + row, 2 * col + 1] != "x":
          states_arr[state] += 1

        for action in range(actions):
          new_row, new_col = row, col
          done = False

          if action == 0:
            new_row = min(row + 1, row_limit)
          if action == 1:
            new_row = max(row - 1, 0)
          if action == 2:
            new_col = min(col + 1, col_limit)
          if action == 3:
            new_col = max(col - 1, 0)

          if self.desc[1 + new_row, 2 * new_col + 1] == b"C":
            reward = -100
          elif self.desc[1 + new_row, 2 * new_col + 1] == b"P":
            reward = -1000
          elif self.desc[1 + new_row, 2 * new_col + 1] == b"x":
            reward = 500
            done = True
          else:
            reward = 100

          new_state = self.encode(new_row, new_col)
          P[state][action].append((1.0, new_state, reward, done))

    states_arr /= states_arr.sum()
    discrete.DiscreteEnv.__init__(self, states, actions, P, states_arr)


  def encode(self, car_row, car_col):
    return car_row * 10 + car_col

  def decode(self, i):
    out = []
    out.append(i % 10)
    i = i // 10
    out.append(i)
    assert 0 <= i < 10
    return reversed(out)

  def render(self, mode="human"):
    outfile = StringIO() if mode == "ansi" else sys.stdout

    out = self.desc.copy().tolist()
    out = [[c.decode("utf-8") for c in line] for line in out]
    car_row, car_col = self.decode(self.s)

    ul = lambda x: "_" if x == " " else x

    if self.desc[1 + car_row, 2 * car_col + 1] != b"x":
      out[1 + car_row][2 * car_col + 1] = utils.colorize(
        out[1 + car_row][2 * car_col + 1], "red", highlight=True
      )
    else:
      out[1 + car_row][2 * car_col + 1] = utils.colorize(
        ul(out[1 + car_row][2 * car_col + 1]), "green", highlight=True
      )

    outfile.write("\n".join(["".join(row) for row in out]) + "\n")

    if self.lastaction is not None:
      outfile.write(
        f"direction: [{['↓', '↑', '→', '←'][self.lastaction]} ]\n"
      )
    else:
      outfile.write("\n")

    if mode != "human":
      with closing(outfile):
        return outfile.getvalue()
