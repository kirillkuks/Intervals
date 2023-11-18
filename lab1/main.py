from __future__ import annotations
import os

from typing import List, Tuple
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

    data2, deltas2 = dataBuilder.load_data(DataSample.kMinus05, 42)
    sample2 = [x_k - delta_k for x_k, delta_k in zip(data2, deltas2)]
    interval_sample2 = dataBuilder.make_intervals(sample2)[start_pos:end_pos]

    # x = [i for i in range(len(deltas))]

    # plt.plot(x, sample2, 'bo')
    # plt.plot(x, data2, 'go')
    # plt.show()

    solver = JaccardSolver()
    solver.plot_intervals([interval_sample1], ['X1'], 'X1', 'X1')
    solver.plot_intervals([interval_sample2], ['X2'], 'X2', 'X2')

    print(f"intervals_x1 Jaccard = {Interval.jaccard_index(interval_sample1)}")
    print(f"intervals_x2 Jaccard = {Interval.jaccard_index(interval_sample2)}")

    solver.plot_intervals(
        [interval_sample1, interval_sample2],
        ['X1', 'X2'],
        'X1 and X2',
        'X1X2')
    
    r = solver.solve(interval_sample1, interval_sample2)
    solver.plot(interval_sample1, interval_sample2, 1000, True)

    solver.plot_sample_moda(interval_sample1, 'X1')
    solver.plot_sample_moda(interval_sample2, 'X2')
    solver.plot_sample_moda(
        Interval.combine_intervals(interval_sample1, Interval.scale_intervals(interval_sample2, r)),
        'X1 union R_opt X2', 'X1RX2')
    
    solver.plot_moda_r(interval_sample1, interval_sample2, 75)

    inner_est = solver.find_r_est(interval_sample1, interval_sample2, 'inner', 1000, 0.95)
    print(f'inner est = {inner_est.to_str()}')
    outer_est = solver.find_r_est(interval_sample1, interval_sample2, 'outer', 75, 0.95)
    print(f'outer est = {outer_est.to_str()}')
    solver.plot_inner_outer_estimations(interval_sample1, interval_sample2, 75, True, r, inner_est, outer_est, 0.95)

    solver.plot_intervals(
        [interval_sample1, Interval.scale_intervals(interval_sample2, r)],
        ['X1', 'R_opt * X2'],
        'X1 union R_opt * X2',
        'X1RX2',
        False)
    
    return


if __name__ == '__main__':
    main()
