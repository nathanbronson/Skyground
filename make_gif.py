import imageio

from tqdm import tqdm
import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")

from ridge_map import RidgeMap, us_bounds
from api_utils import get_timed_paths, freeze_oob, ALT_FAC

VERTICAL_RATIO = 50#30#25#50

if __name__ == "__main__":
    _box = (-125, 24, -66, 49)
    box = _box[1::2] + _box[::2]
    _box = (-129, 15, -56, 56)
    rm = RidgeMap(_box)
    values = rm.get_elevation_data(num_lines=350)
    __times, _ = get_timed_paths(box=box, only_first=None, alt_fac=ALT_FAC, expire=3600, factor=2)
    _times = freeze_oob(__times, box=box, timed=True)
    def make_plot(values, flights, path=None):
        _v = rm.preprocess(values=values, lake_flatness=1, water_ntile=37, vertical_ratio=VERTICAL_RATIO, bounds=us_bounds)
        ax = rm.plot_map_with_extra_lines(
            values=_v,
            extra_lines=flights,
            no_label=True,
            experimental_zorder=True,
            line_color=plt.get_cmap("spring"),
            background_color=(33/255, 41/255, 70/255),
            linewidth=3,
            plane_width=4
        )
        plt.style.use("cyberpunk")
        mplcyberpunk.make_lines_glow(ax)
        if path is not None:
            plt.savefig(path, bbox_inches='tight')
        return ax
    def make_gif(values, _t, duration=10):
        filenames = []
        for n, t in tqdm(list(enumerate(_t))):
            filenames.append("./gif/{}.png".format(n))
            make_plot(values, t, path=filenames[-1])
            plt.close()
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('./flights.gif', images, duration=duration/len(filename))
    make_gif(values, _times[::100])