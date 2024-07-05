import os
from time import time, sleep

import matplotlib.pyplot as plt
import mplcyberpunk
plt.style.use("cyberpunk")

from ridge_map import RidgeMap, us_bounds
from api_utils import get_paths_at_time, freeze_oob, ALT_FAC

SCRIPT = """osascript -e 'tell application "System Events"
tell every desktop
  set picture to "{}"
end tell
end tell'
"""

NUM_PLANES = 500
UPDATE_INTERVAL = 6 * 60
STATE_INTERVAL = 5
INTERVAL = 3600
VERTICAL_RATIO = 50
PIC_DIR = os.path.expanduser("~/.skyground/")
file = PIC_DIR + "skyground_{}.png"

if __name__ == "__main__":
    if not os.path.isdir(PIC_DIR):
        os.system("mkdir {}".format(PIC_DIR))
    _box = (-125, 24, -66, 49)
    box = _box[1::2] + _box[::2]
    _box = (-129, 15, -56, 56)
    rm = RidgeMap(_box)
    values = rm.get_elevation_data(num_lines=350)
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
    _cur_paths, crafts = get_paths_at_time(int(time()), box=box, alt_fac=ALT_FAC, only_first=NUM_PLANES, expire=INTERVAL, factor=2)
    cur_paths = freeze_oob(_cur_paths, box=box)
    figalt = True
    last_time = 0
    failed = False
    i = 0
    while True:
        i += 1
        if time() - last_time >= UPDATE_INTERVAL:
            if not failed:
                last_time = time()
                ax = make_plot(values, cur_paths, file.format(int(figalt)))
                plt.close()
                sleep(1)
                os.system(SCRIPT.format(file.format(int(figalt))))
                figalt = not figalt
                os.system("rm {}".format(file.format(int(figalt))))
            try:
                _cur_paths, crafts = get_paths_at_time(int(time()), box=box, alt_fac=ALT_FAC, only_first=NUM_PLANES, expire=INTERVAL, crafts=crafts, state=i%STATE_INTERVAL==0, factor=2)
                cur_paths = freeze_oob(_cur_paths, box=box)
                failed = False
            except Exception as e:
                print("failure", type(e), e)
                failed = True
        else:
            sleep(5)