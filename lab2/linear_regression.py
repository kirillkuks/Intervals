from __future__ import annotations

import numpy as np
from typing import List, Tuple
from scipy.optimize import linprog
from numpy.typing import ArrayLike
from shapely.geometry import Polygon
from interval import Interval
from matplotlib import pyplot as plt
from math import inf


def img_save_dst() -> str:
    return 'doc\\img\\'


class LinearRegression:
    def __init__(self, x: List[float], y: List[Interval]) -> None:
        assert len(x) == len(y)

        self.x: List[float] = x.copy()
        self.y: List[Interval] = y.copy()
        self.size = len(x)
        self.params_num = 2
        self.regression_params: List[float] = None
        self.inform_set: Polygon = None

    def build_point_regression(self) -> List[float]:
        if self.regression_params is not None:
            return self.regression_params

        c = np.array([0 for _ in range(self.params_num)] + [1 for _ in range(self.size)])

        A_ub: ArrayLike[ArrayLike[float]] = np.array([np.array([0.0 for _ in range(2 + self.size)]) for _ in range(2 * self.size)])
        b_ub: ArrayLike[float] = np.array([0.0 for _ in range(2 * self.size)])

        for i, (x_i, y_i) in enumerate(zip(self.x, self.y)):
            A_ub[2 * i][0], A_ub[2 * i][1] = -1, -x_i
            A_ub[2 * i + 1][0], A_ub[2 * i + 1][1] = 1, x_i

            A_ub[2 * i][i + 2] = -y_i.rad()
            A_ub[2 * i + 1][i + 2] = -y_i.rad()

            b_ub[2 * i] = -y_i.mid()
            b_ub[2 * i + 1] = y_i.mid()

        bounds = np.array([(None, None) for _ in range(self.params_num)] + [(0, None) for _ in range(self.size)])

        res = linprog(method='simplex', c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        self.regression_params = [res.x[i] for i in range(self.params_num)]
        print(f'b = {self.regression_params}')
        return self.regression_params

    def build_inform_set(self) -> Polygon:
        if self.inform_set is not None:
            return self.inform_set
        
        lower, upper = -10.0, 10.0
        self.inform_set = self._create_codition_band(0, lower, upper)

        for i in range(1, self.size):
            self.inform_set = self.inform_set.intersection(self._create_codition_band(i, lower, upper))

        print(self.inform_set.exterior.xy)
        return self.inform_set

        
    def _create_codition_band(self, condition_idx: int, lower: float, upper: float) -> Polygon:
        assert 0 <= condition_idx < self.size
        
        x_i = self.x[condition_idx]
        y_i = self.y[condition_idx]

        return Polygon((
            (lower, -lower * x_i + y_i.left),
            (lower, -lower * x_i + y_i.right),
            (upper, -upper * x_i + y_i.right),
            (upper, -upper * x_i + y_i.left)
        ))


class Plotter:
    class Point:
        def __init__(self, x: float, y: float, label: str = '') -> None:
            self.x = x
            self.y = y
            self.label = label


    def __init__(self) -> None:
        pass

    def plot_sample(self, x: List[float], y: List[Interval], show: bool = False, title: str='') -> None:
        for x_i, y_i in zip(x, y):
            plt.plot((x_i, x_i), (y_i.left, y_i.right), 'b')

        if show:
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(title)
            plt.savefig(f'{img_save_dst()}{title}.png', dpi=200)
            plt.clf()

    def plot_inform_set(self, inform_set: Polygon, points: List[Plotter.Point] = [], title: str = '') -> None:
        plt.plot(*inform_set.exterior.xy, label='inform set edge')

        for point in points:
            plt.plot(point.x, point.y, 'o', label=point.label)

        aabb = [inf, inf, -inf, -inf] # b1_min, b0_min, b1_max, b0_max

        for b1, b0 in zip(inform_set.exterior.xy[0], inform_set.exterior.xy[1]):
            aabb[0], aabb[2] = min(aabb[0], b1), max(aabb[2], b1)
            aabb[1], aabb[3] = min(aabb[1], b0), max(aabb[3], b0)

        self._plot_aabb(aabb)

        plt.xlabel('beta1')
        plt.ylabel('beta0')
        plt.title(f'Inform set {title}')
        plt.legend(loc='upper right')
        plt.savefig(f'{img_save_dst()}InformSet{title}.png', dpi=200)
        plt.clf()

    def plot(self, regression: LinearRegression, title: str = '') -> None:
        self.plot_sample(regression.x, regression.y)

        params = regression.build_point_regression()
        plt.plot(
            regression.x,
            [params[0] + params[1] * x for x in regression.x],
            'r',
            label=f'y = {round(params[0], 4)} + {round(params[1], 4)}x',
            linewidth=1.0
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Point Regression {title}')
        plt.legend()
        plt.savefig(f'{img_save_dst()}PointRegression{title}.png', dpi=200)
        plt.clf()
    
    def plot_corridor(self,
                      regression: LinearRegression,
                      predict: bool = False,
                      pos_x_predict_size = 5,
                      neg_x_predict_size = 5,
                      title: str = ''
                      ) -> None:
        self.plot_sample(regression.x, regression.y)

        y_min, y_max = [], []
        xs = []

        predict_delta = 0.25

        if predict:
            x = regression.x[0] - predict_delta * neg_x_predict_size
            while x < regression.x[0]:
                mi, ma = self._find_min_max_edges_in_corridor(x, regression.inform_set)

                xs.append(x)
                y_min.append(mi)
                y_max.append(ma)

                x += predict_delta

        for x in regression.x:
            mi, ma = self._find_min_max_edges_in_corridor(x, regression.inform_set)

            xs.append(x)
            y_min.append(mi)
            y_max.append(ma)

        if predict:
            i = 0
            x = regression.x[-1]
            while i < predict_delta * pos_x_predict_size:
                mi, ma = self._find_min_max_edges_in_corridor(x + i, regression.inform_set)

                xs.append(x + i)
                y_min.append(mi)
                y_max.append(ma)

                i += predict_delta

        plt.fill_between(xs, y_min, y_max, alpha=0.5, label='inform set corridor')

        params = regression.build_point_regression()
        plt.plot(
            xs,
            [params[0] + params[1] * x for x in xs],
            'r',
            label=f'y = {round(params[0], 4)} + {round(params[1], 4)}x',
            linewidth=1.0
            )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Imform set corridor {title}')
        plt.legend()
        plt.savefig(f'{img_save_dst()}InformSetCorridor{title}.png', dpi=200)
        plt.clf()

    def _find_min_max_edges_in_corridor(self, x: float, inform_set: Polygon) -> Tuple[float, float]:
        mi, ma = inf, -inf

        for b1, b0 in zip(inform_set.exterior.xy[0], inform_set.exterior.xy[1]):
            y = b0 + b1 * x

            mi = min(mi, y)
            ma = max(ma, y)

        return mi, ma
    
    def _plot_aabb(self, aabb: List[float]) -> None:
        assert len(aabb) == 4

        plt.plot((aabb[0], aabb[0]), (aabb[1], aabb[3]), '--r')
        plt.plot((aabb[0], aabb[2]), (aabb[3], aabb[3]), '--r')
        plt.plot((aabb[2], aabb[2]), (aabb[3], aabb[1]), '--r')
        line, = plt.plot((aabb[2], aabb[0]), (aabb[1], aabb[1]), '--r')
        line.set_label('bounding box')
