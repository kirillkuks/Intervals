import os

from typing import List, Tuple

from numpy import random as rnd

from interval import Interval
from linear_regression import LinearRegression, Plotter, RemainderAnalyzer


def is_float(value: str) -> bool:
    if value is None:
        return False
    
    try:
        float(value)
        return True
    except:
        return False
    

def median(sample: List[float]) -> float:
    return sorted(sample)[(len(sample) >> 1) + 1]


class DataSample:    
    def __init__(self, factors: List[float]) -> None:
        self.factors = factors.copy()

    def factors_num(self) -> int:
        return len(self.factors)
    
    def factor_name(self, factor_idx) -> str:
        assert 0 <= factor_idx < self.factors_num()
        return f'{round(self.factors[factor_idx], 2)}V' 


class IntervalDataBuilder:
    def __init__(self, data_sample: DataSample, working_dir: str) -> None:
        self.working_dir = working_dir
        self.data_sample = data_sample
        self.rnd = rnd.default_rng(42)

    def get_eps(self) -> float:
        return 25.0 # 1 << 9 ?
        
    def create_interval_sample(self, min_max: bool = True) -> List[Interval]:
        data_files = os.listdir(self.working_dir)
        responses = []

        for i in range(self.data_sample.factors_num()):
            data_name = self.data_sample.factor_name(i)
            used_data_file: str = None

            for data_file in data_files:
                if data_name in data_file and data_file.index(data_name) == 0:
                    used_data_file = data_file
                    break

            assert used_data_file is not None
            
            data, deltas = self.load_samples(used_data_file)
            sample = [x_k - d_k for x_k, d_k in zip(data, deltas)]

            if min_max:
                mi = min(sample)
                ma = max(sample)
                responses.append(Interval(mi, ma))
            else:
                med = median(sample)
                responses.append(Interval(med - self.get_eps(), med + self.get_eps()))

        return responses

    def load_data(self, file_name: str) -> List[float]:
        with open(f'{self.working_dir}\\{file_name}') as f:
            stop_position = file_name[file_name.rindex('_sp') + 3:]
            stop_position = int(stop_position[:stop_position.rindex('.dat')])

            data = []
            for fileline in f.readlines():
                numbers = fileline.split(' ')
                floats = [float(number) for number in numbers if is_float(number)]

                if (len(floats) > 2):
                    data.append(floats[1])
            
            data = data[stop_position:] + data[:stop_position]
            return data
        
    def load_samples(self, file_name: str) -> Tuple[List[float], List[float]]:
        return self.load_data(file_name), self.load_data('0.0V_sp443.dat')
        
    def make_intervals(self, point_sample: List[float]) -> List[Interval]:
        eps = self.get_eps()
        return [Interval(x - eps, x + eps) for x in point_sample]


def main():
    base_dir = os.getcwd()
    data_dir = base_dir[:base_dir.rindex('\\')] + '\\data\\dataset2'

    factors = [-0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45]
    data_sample = DataSample(factors)

    data_builder = IntervalDataBuilder(data_sample, data_dir)
    responses1 = data_builder.create_interval_sample(True)
    responses2 = data_builder.create_interval_sample(False)

    for i, responses in enumerate([responses1, responses2]):
        sample_name = f'X{i + 1}'
        remainder_name = f'E{i + 1}'

        regression = LinearRegression(data_sample.factors, responses)
        regression.build_point_regression()
        regression.build_inform_set()

        remainder_analyzer = RemainderAnalyzer()
        remainders = remainder_analyzer.build_remainders(data_sample.factors, responses, regression)
        l = remainder_analyzer.get_high_leverage(regression)
        r = remainder_analyzer.get_relative_residual(regression)

        # regression = LinearRegression(data_sample.factors, remainders)
        # regression.build_point_regression()
        # regression.build_inform_set()

        plotter = Plotter(True)
        plotter.plot_sample(regression.x, regression.y, True, sample_name)
        plotter.plot_sample(regression.x, remainders, True, remainder_name)
        plotter.plot(regression, sample_name)
        plotter.plot_inform_set(
            regression.inform_set,
            [Plotter.Point(regression.regression_params[1], regression.regression_params[0], 'point regression')],
            sample_name)
        plotter.plot_corridor(regression, False, title=sample_name)
        plotter.plot_status_diagram(regression, False, sample_name)
        plotter.plot_status_diagram(regression, True, f'Zoom{sample_name}')

    return

if __name__ == '__main__':
    main()
