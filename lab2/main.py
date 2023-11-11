import os
from typing import List
from matplotlib import pyplot as plt
from interval import Interval
from linear_regression import LinearRegression, Plotter
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
        self.rnd = rnd.default_rng()

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


def main():
    working_dir = os.getcwd()
    working_dir = working_dir[:working_dir.rindex('\\')]
    database_dir = working_dir + '\\data\\dataset1'
    working_dir = database_dir + '\\+0_5V'

    dataBuilder = IntervalDataBuilder(working_dir)
    intervals_y1 = dataBuilder.load_interval_sample('+0_5V_85.txt')
    intervals_y2 = Interval.expand_intervals(intervals_y1, 0.05)

    x = [i for i in range(len(intervals_y1))]

    for interval_responses, sample_name in zip([intervals_y1, intervals_y2], ['X1', 'X2']):
        print(f'Jaccard Index of {sample_name}: {Interval.jaccard_index(interval_responses)}')

        regression = LinearRegression(x, interval_responses)
        regression.build_point_regression()
        regression.build_inform_set()

        plotter = Plotter()
        plotter.plot_sample(x, interval_responses, True, sample_name)
        plotter.plot(regression, sample_name)

        plotter.plot_corridor(regression, predict=True, title=sample_name)

        points = [
            Plotter.Point(regression.regression_params[1], regression.regression_params[0], 'point regression')
            ]
        plotter.plot_inform_set(regression.inform_set, points, sample_name)


if __name__ == '__main__':
    main()