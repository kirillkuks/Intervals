from typing import List, Tuple
from matplotlib import pyplot as plt
from interval import Interval


def img_save_dst() -> str:
    return 'doc\\img\\'


class JaccardSolver:
    def __init__(self) -> None:
        self.x_1 : List[Interval] = []
        self.x_2 : List[Interval]= []
        pass

    def solve(self, intervals1: List[Interval], intervals2: List[Interval]) -> float:
        self.x_1, self.x_2 = intervals1, intervals2
        r_1, r_2 = self._find_edges()
        # print(f'r1 = {r_1}, r2 = {r_2}')

        while r_2 - r_1 > 1e-9:
            r = (r_1 + r_2) * 0.5
            size = r_2 - r_1
            r1, r2 = r - size * 0.01, r + size * 0.01

            new_jaccard_r1 = Interval.jaccard_index(self._build_sample(r1))
            new_jaccard_r2 = Interval.jaccard_index(self._build_sample(r2))

            if new_jaccard_r1 > new_jaccard_r2:
                r_2 = r2
            else:
                r_1 = r1


        r = (r_1 + r_2) * 0.5
        print(f'R = {r}, Jaccard Index = {Interval.jaccard_index(self._build_sample(r))}')
        return r
    
    def find_r_est(
            self,
            intervals1: List[Interval],
            intervals2: List[Interval],
            est: str,
            size: int,
            alpha: float
            ) -> Interval:
        assert 0.0 < alpha < 1.0
        self.x_1, self.x_2 = intervals1, intervals2
        r_1, r_2 = self._find_edges()
        r_1, r_2 = r_1 - 0.1, r_2 + 0.1

        width = r_2 - r_1
        delta = width / size

        x = [r_1 + i * delta for i in range(size)]
        r = self.solve(intervals1, intervals2)

        y = []
        edge = 0

        if est == 'inner':
            y = [Interval.jaccard_index(self._build_sample(x_k)) for x_k in x]
            edge = Interval.jaccard_index(self._build_sample(r)) * alpha
        elif est == 'outer':
            y = [Interval.find_moda(self._build_sample(x_k))[0] for x_k in x]
            edge = Interval.find_moda(self._build_sample(r))[0] * alpha
        else:
            assert False

        r_min, r_max = None, None

        for i, (x_k, y_k) in enumerate(zip(x, y)):
            if r_min is None and y_k > edge:
                r_min = x_k
            
            if r_min is not None and r_max is None and y_k < edge:
                r_max = x[i - 1]

        return Interval(r_min, r_max)


    def plot(self, intervals1: List[Interval], intervals2: List[Interval], size: int, draw_max: bool = False) -> None:
        self.x_1, self.x_2 = intervals1, intervals2
        r_1, r_2 = self._find_edges()
        r_1, r_2 = r_1 - 0.5, r_2 + 0.5
        width = r_2 - r_1
        delta = width / size

        x = [r_1 + i * delta for i in range(size)]
        y = [Interval.jaccard_index(self._build_sample(x_k)) for x_k in x]

        plt.plot(x, y)

        if draw_max:
            r = self.solve(self.x_1, self.x_2)
            jaccard = Interval.jaccard_index(self._build_sample(r))
            plt.plot(r, jaccard, 'ro', markersize=12, label=f'optimal R = {round(r, 3)}', alpha=0.5)
            plt.legend()

        plt.title(r'Jaccard index of X_1 union R * X_2')
        plt.xlabel('R')
        plt.ylabel('Jaccard index')
        plt.savefig(f'{img_save_dst()}_Jaccard.png')
        plt.clf()

    def plot_inner_outer_estimations(
            self,
            intervals1: List[Interval],
            intervals2: List[Interval],
            size: int,
            normalize: bool = False,
            r: float = 0.0,
            inner_est: Interval = None,
            outer_est: Interval = None,
            alpha = 1.0
            ) -> None:
        self.x_1, self.x_2 = intervals1, intervals2
        r_1, r_2 = self._find_edges()
        r_1, r_2 = r_1 - 0.1, r_2 + 0.1
        width = r_2 - r_1
        delta = width / size

        normalize_coef = 1.0
        max_jaccard = Interval.jaccard_index(self._build_sample(r))

        if normalize:
            max_moda = len(intervals1) + len(intervals2)
            normalize_coef = max_jaccard / float(max_moda)

        x = [r_1 + i * delta for i in range(size)]
        y = [Interval.find_moda(self._build_sample(x_k))[0] * normalize_coef for x_k in x]
        y1 = [Interval.jaccard_index(self._build_sample(x_k)) for x_k in x]
            
        plt.plot(x, y, label='intervals in moda')
        plt.plot(x, y1, label='jaccard index')

        if inner_est is not None:
            fig = None
            for r_est in [inner_est.left, inner_est.right]:
                fig, = plt.plot((r_est, r_est), (0, Interval.jaccard_index(self._build_sample(r_est))), 'r--')
            
            fig.set_label('inner R estimation')

        if outer_est is not None:
            fig = None
            for r_est in [outer_est.left, outer_est.right]:
                fig, = plt.plot((r_est, r_est), (0, Interval.find_moda(self._build_sample(r_est))[0] * normalize_coef), 'g--')

            fig.set_label('outer R estimation')

        max_jaccard *= alpha
        plt.plot((r_1, r_2), (max_jaccard, max_jaccard), 'k--', label=f'trust level, alpha = {alpha}')
        print(f'normalize coef = {normalize_coef}')

        plt.title('Inner and outer R estimations')
        plt.legend()
        plt.savefig(f'{img_save_dst()}_InnerOuter.png')
        plt.clf()

    def plot_moda_r(self, intervals1: List[Interval], intervals2: List[Interval], size: int):
        self.x_1, self.x_2 = intervals1, intervals2
        r_1, r_2 = self._find_edges()
        r_1, r_2 = r_1 - 0.25, r_2 + 0.25
        width = r_2 - r_1
        delta = width / size

        x = [r_1 + i * delta for i in range(size)]
        y = [Interval.find_moda(self._build_sample(x_k))[0] for x_k in x]
            
        plt.plot(x, y)
        plt.xlabel('R')
        plt.ylabel('intervals num in moda')
        plt.title('Intervals num in X1 union R * X2')
        plt.savefig(f'{img_save_dst()}_ModaR.png')
        plt.clf()

    def plot_sample_moda(
            self,
            intervals: List[Interval],
            title: str = '',
            short_name: str = None
            ) -> None:
        _, moda_hist, moda_intervals, moda = Interval.find_moda(intervals)

        x, y = [], []

        for bar_size, interval in zip(moda_hist, moda_intervals):
            x.extend([interval.left, interval.right])
            y.extend([bar_size, bar_size])

        print(f'Moda for {title}: {moda.to_str()}, wid = {moda.wid()}')

        plt.plot(x, y)
        plt.xlabel('data')
        plt.ylabel('intervals in intersection')
        plt.title(f'moda {title} hist')
        plt.savefig(f'{img_save_dst()}_Moda{short_name if short_name is not None else title}Hist.png')
        plt.clf()

    def plot_intervals(
            self,
            array_intervals: List[List[Interval]],
            labels: List[str],
            title: str,
            save_name: str,
            reset: bool = True
            ) -> None:
        styles = ['b', 'g', 'r', 'y', 'k']
        start = 0

        for j, intervals in enumerate(array_intervals):
            for i, interval in enumerate(intervals):
                line, = plt.plot((start + i, start + i), (interval.left, interval.right), styles[j % len(styles)])

                if i == 0:
                    line.set_label(labels[j])

            if not reset: 
                start += len(intervals)

        plt.legend()
        plt.title(title)
        plt.savefig(f'{img_save_dst()}_{save_name}.png', dpi=200)
        plt.clf()


    def _build_sample(self, r: float) -> List[Interval]:
        return Interval.combine_intervals(self.x_1, Interval.scale_intervals(self.x_2, r))

    def _find_edges(self) -> Tuple[float, float]:
        min_min_x1 = min([interval.pro().left for interval in self.x_1])
        max_max_x2 = max([interval.pro().right for interval in self.x_2])

        max_max_x1 = max([interval.pro().right for interval in self.x_1])
        min_min_x2 = min([interval.pro().left for interval in self.x_2])

        t1, t2 = min_min_x1 / max_max_x2, max_max_x1 / min_min_x2
        return min(t1, t2), max(t1, t2)
