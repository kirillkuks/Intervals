import os
from typing import List, Tuple
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


class DataSample:
    kPlus05 = 0,
    kMinus05 = 1,
    kZero = 2,

    _kDict = {
        kPlus05: '+0_5V',
        kMinus05: '-0_5V',
        kZero: 'ZeroLine'
    }

    @staticmethod
    def to_str(data_sample: int) -> str:
        return DataSample._kDict[data_sample]


class IntervalDataBuilder:
    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir
        self.rnd = rnd.default_rng(42)

    def get_eps(self) -> float:
        return self.rnd.uniform(0.01, 0.05)

    def load_sample(self, filename: str) -> List[float]:
        with open(f'{self.working_dir}\\{filename}') as f:
            stop_position_str = f.readline()
            stop_position = int(stop_position_str[stop_position_str.index('=') + 1:])

            deltas = []
            for fileline in f.readlines():
                numbers = fileline.split(' ')
                floats = [float(number) for number in numbers if is_float(number)]

                deltas.append(floats[1])
            
            stop_position = len(deltas) - stop_position
            deltas = deltas[stop_position:] + deltas[:stop_position]
            return deltas
        
    def load_data(self, data_sample: DataSample, sample_idx: int) -> Tuple[List[float], List[float]]:
        data_subdir_name = DataSample.to_str(data_sample)
        data = self.load_sample(f'{data_subdir_name}\\{data_subdir_name}_{sample_idx}.txt')

        deltas_subdir_name = DataSample.to_str(DataSample.kZero)
        deltas = self.load_sample(f'{deltas_subdir_name}\\{deltas_subdir_name}_{sample_idx}.txt')

        return data, deltas
        
    def make_intervals(self, point_sample: List[float]) -> List[Interval]:
        eps = 1.0 / (1 << 14) * 100.0
        return [Interval(x - eps, x + eps) for x in point_sample]


def main():
    working_dir = os.getcwd()
    working_dir = working_dir[:working_dir.rindex('\\')]
    database_dir = working_dir + '\\data\\dataset1'

    dataBuilder = IntervalDataBuilder(database_dir)

    start_pos, end_pos = 500, 700

    data, deltas = dataBuilder.load_data(DataSample.kPlus05, 0)
    sample = [x_k - delta_k for x_k, delta_k in zip(data, deltas)]
    interval_sample1 = dataBuilder.make_intervals(sample)[start_pos:end_pos]
    interval_sample2 = dataBuilder.make_intervals(data)[start_pos:end_pos]

    x = [i for i in range(len(interval_sample1))]

    for interval_responses, sample_name in zip([interval_sample1, interval_sample2], ['X1', 'X2']):
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
