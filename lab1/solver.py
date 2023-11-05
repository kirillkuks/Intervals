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
        print(f'r1 = {r_1}, r2 = {r_2}')

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
            plt.plot(r, jaccard, 'ro', markersize=12, label=f'optimal R = {round(r, 3)}')
            plt.legend()


        plt.title(r'Jaccard index of X_1 union R * X_2')
        plt.xlabel('R')
        plt.ylabel('Jaccard index')
        plt.savefig(f'{img_save_dst()}Jaccard.png')
        plt.clf()


    def plot_intervals(
            self,
            array_intervals: List[List[Interval]],
            labels: List[str],
            title: str,
            save_name: str,
            reset: bool = True) -> None:
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
        plt.savefig(f'{img_save_dst()}{save_name}.png')
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
