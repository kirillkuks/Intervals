import os

from typing import List
from matplotlib import pyplot as plt
from interval import Interval
from solver import JaccardSolver
from numpy import random as rnd


def is_float(value: str) -> bool:
    if value is None:
        return False
    
    try:
        float(value)
        return True
    except:
        return False



class IntervalDataBuilder:
    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir
        self.rnd = rnd.default_rng(42)

    def get_eps(self) -> float:
        return self.rnd.uniform(0.01, 0.05)

    def load_interval_sample(self, filename: str) -> List[Interval]:
        intervals = []
        x = []

        with open(self.working_dir + '\\' + filename) as f:
            for i, fileline in enumerate(f.readlines()):
                if i < 1:
                    continue
                if i > 100:
                    break

                numbers = fileline.split(' ')
                floats = [float(number) for number in numbers if is_float(number)]

                x.append(floats[1])
                center = floats[1]
                eps = self.get_eps()
                intervals.append(Interval(center - eps, center + eps, True))

        return intervals
    
    
    def change_workin_dir(self, new_working_dir: str) -> None:
        self.working_dir = new_working_dir

    def find_biggest_jaccard(self) -> None:
        file_base_name = self.working_dir[self.working_dir.rindex('\\') + 1:]

        i = 0
        biggest_idx = 0
        biggest_jaccard = 0
        while True:
            new_filename = file_base_name + f'_{i}.txt'

            if os.path.isfile(self.working_dir + "\\" + new_filename) == False:
                break

            intervals_data = self.load_interval_sample(new_filename)
            jaccard = Interval.jaccard_index(intervals_data)
            if jaccard > biggest_jaccard:
                biggest_jaccard = jaccard
                biggest_idx = i

            i += 1

        return biggest_idx


def main():
    working_dir = os.getcwd()
    working_dir = working_dir[:working_dir.rindex('\\')]
    database_dir = working_dir + '\\data\\dataset1'
    working_dir = database_dir + '\\+0_5V'

    dataBuilder = IntervalDataBuilder(working_dir)
    intervals_x1 = dataBuilder.load_interval_sample('+0_5V_85.txt')
    dataBuilder.change_workin_dir(database_dir + '\\-0_5V')
    intervals_x2 = dataBuilder.load_interval_sample('-0_5V_6.txt')

    #intervals_x1 = Interval.expand_intervals(intervals_x1, 0.05)
    #intervals_x2 = Interval.expand_intervals(intervals_x2, 0.05)

    print(f"intervals_x1 Jaccard = {Interval.jaccard_index(intervals_x1)}")
    print(f"intervals_x2 Jaccard = {Interval.jaccard_index(intervals_x2)}")

    solver = JaccardSolver()
    solver.plot_intervals(
        [intervals_x1, intervals_x2],
        ['X1', 'X2'],
        'X1 and X2',
        'X1X2')

    r = solver.solve(intervals_x1, intervals_x2)
    solver.plot(intervals_x1, intervals_x2, 1000, True)

    inner_est = solver.find_r_est(intervals_x1, intervals_x2, 'inner', 1000, 0.95)
    outer_est = solver.find_r_est(intervals_x1, intervals_x2, 'outer', 100, 0.95)

    solver.plot_sample_moda(intervals_x1, 'X1')
    solver.plot_sample_moda(intervals_x2, 'X2')

    solver.plot_moda_r(intervals_x1, intervals_x2, 100)
    solver.plot_inner_outer_estimations(intervals_x1, intervals_x2, 100, True, r, inner_est, outer_est, 0.95)

    solver.plot_sample_moda(
        Interval.combine_intervals(intervals_x1, Interval.scale_intervals(intervals_x2, r)),
        'X1 union R_opt X2', 'X1RX2')

    solver.plot_intervals(
        [intervals_x1, Interval.scale_intervals(intervals_x2, r)],
        ['X1', 'R_opt * X2'],
        'X1 union R_opt * X2',
        'X1RX2',
        False)

    return


if __name__ == '__main__':
    main()
