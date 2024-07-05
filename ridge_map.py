#THANKS TO https://github.com/ColCarroll/ridge_map
#THIS CODE COMES FROM THERE WITH SIGNIFICANT MODIFICATIONS
"""3D maps with 1D lines."""

from urllib.request import urlopen
from tempfile import NamedTemporaryFile
from json import loads

from matplotlib.collections import LineCollection
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import rank
from skimage.morphology import square
from skimage.util import img_as_ubyte
from shapely.geometry import shape, Point
import geopandas as gpd
import osmnx as ox
import pandas as pd
from tqdm import tqdm

import srtm
import mplcyberpunk
plt.style.use("cyberpunk")

with open("./us_boundaries.json", "r") as doc:
    us_bounds = shape(loads(doc.read())["features"][0]["geometry"])

class FontManager:
    """Utility to load fun fonts from https://fonts.google.com/ for matplotlib.

    Find a nice font at https://fonts.google.com/, and then get its corresponding URL
    from https://github.com/google/fonts/

    Use like:

    fm = FontManager()
    fig, ax = plt.subplots()

    ax.text("Good content.", fontproperties=fm.prop, size=60)
    """

    def __init__(
        self,
        github_url="https://github.com/google/fonts/raw/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf",  # pylint: disable=line-too-long
    ):
        """
        Lazily download a font.

        Parameters
        ----------
        github_url : str
            Can really be any .ttf file, but probably looks like
            "https://github.com/google/fonts/raw/main/ofl/cinzel/Cinzel%5Bwght%5D.ttf" # pylint: disable=line-too-long
        """
        self.github_url = github_url
        self._prop = None

    @property
    def prop(self):
        """Get matplotlib.font_manager.FontProperties object that sets the custom font."""
        if self._prop is None:
            with NamedTemporaryFile(delete=False, suffix=".ttf") as temp_file:
                # pylint: disable=consider-using-with
                temp_file.write(urlopen(self.github_url).read())
                self._prop = fm.FontProperties(fname=temp_file.name)
        return self._prop


class RidgeMap:
    """Main class for interacting with art.

    Keeps state around so no servers are hit too often.
    """

    def __init__(self, bbox=(-71.928864, 43.758201, -70.957947, 44.465151), font=None):
        """Initialize RidgeMap.

        Parameters
        ----------
        bbox : list-like of length 4
            In the form (long, lat, long, lat), describing the
            (bottom_left, top_right) corners of a box.
            http://bboxfinder.com is a useful way to find these tuples.
        font : matplotlib.font_manager.FontProperties
            Optional, a custom font to use. Defaults to Cinzel Regular.
        """
        self.bbox = bbox
        self._srtm_data = srtm.get_data()
        if font is None:
            font = FontManager().prop
        self.font = font
        self.mask = None

    @property
    def lats(self):
        """Left and right latitude of bounding box."""
        return (self.bbox[1], self.bbox[3])

    @property
    def longs(self):
        """Bottom and top longitude of bounding box."""
        return (self.bbox[0], self.bbox[2])

    def get_elevation_data(self, num_lines=80, elevation_pts=300, viewpoint="south"):
        """Fetch elevation data and return a numpy array.

        Parameters
        ----------
        num_lines : int
            Number of horizontal lines to draw
        elevation_pts : int
            Number of points on each line to request. There's some limit to
            this that srtm enforces, but feel free to go for it!
        viewpoint : str in ["south", "west", "north", "east"] (default "south")
            The compass direction from which the map will be visualised.

        Returns
        -------
        np.ndarray
        """
        if viewpoint in ["east", "west"]:
            num_lines, elevation_pts = elevation_pts, num_lines
        values = self._srtm_data.get_image(
            (elevation_pts, num_lines), self.lats, self.longs, 5280, mode="array"
        )
        self._srtm_data = None

        switch = {"south": 0, "west": 3, "north": 2, "east": 1}
        rotations = switch[viewpoint]
        values = np.rot90(m=values, k=rotations)
        return values
    
    def build_bound_mask(self, values, bounds, verbose=True):
        if verbose:
            _tqdm = tqdm
        else:
            _tqdm = lambda x: x
        inverse_lat = lambda _x: ((_x/values.shape[0])*(self.lats[0] - self.lats[1])) + self.lats[1]
        inverse_long = lambda _x: ((_x/values.shape[1])*(self.longs[0] - self.longs[1])) + self.longs[1]
        lats = inverse_lat(np.arange(values.shape[0]))
        longs = inverse_long(np.arange(values.shape[1]))
        lats, longs = np.meshgrid(lats, longs)
        geometry = bounds
        geometry_cut = ox.utils_geo._quadrat_cut_geometry(geometry, quadrat_width=5)
        gdf_nodes = gpd.GeoDataFrame(data={'x':longs.reshape((-1,)), 'y':lats.reshape((-1,))})
        gdf_nodes.name = 'nodes'
        gdf_nodes['geometry'] = gdf_nodes.apply(lambda row: Point((row['x'], row['y'])), axis=1)
        sindex = gdf_nodes.sindex
        points_within_geometry = pd.DataFrame()
        for poly in _tqdm(geometry_cut.geoms):
            # buffer by the <1 micron dist to account for any space lost in the quadrat cutting
            # otherwise may miss point(s) that lay directly on quadrat line
            poly = poly.buffer(1e-14).buffer(0)

            # find approximate matches with r-tree, then precise matches from those approximate ones
            possible_matches_index = list(sindex.intersection(poly.bounds))
            possible_matches = gdf_nodes.iloc[possible_matches_index]
            precise_matches = possible_matches[possible_matches.intersects(poly)]
            points_within_geometry = pd.concat([points_within_geometry, precise_matches])#points_within_geometry = points_within_geometry.append(precise_matches)
        points_within_geometry = points_within_geometry.drop_duplicates(subset=['x', 'y'])
        points_outside_geometry = gdf_nodes[~gdf_nodes.isin(points_within_geometry)]
        m = np.zeros_like(values)
        for r in _tqdm(range(values.shape[0])):
            for c in range(values.shape[1]):
                if np.any(((points_outside_geometry["x"] == longs.T[r][c]) & (points_outside_geometry["y"] == lats.T[r][c])).values):
                    m[r][-1 - c] = np.nan
        return m

    def preprocess(
        self, *, values=None, water_ntile=10, lake_flatness=3, vertical_ratio=40, bounds=None
    ):
        """Get map data ready for plotting.

        You can do this yourself, and pass an array directly to plot_map. This
        gathers all nan values, the lowest `water_ntile` percentile of elevations,
        and anything that is flat enough, and sets the values to `nan`, so no line
        is drawn. It also exaggerates the vertical scale, which can be nice for flat
        or mountainy areas.

        Parameters
        ----------
        values : np.ndarray
            An array to process, or fetch the elevation data lazily here.
        water_ntile : float in [0, 100]
            Percentile below which to delete data. Useful for coasts or rivers.
            Set to 0 to not delete any data.
        lake_flatness : int
            How much the elevation can change within 3 squares to delete data.
            Higher values delete more data. Useful for rivers, lakes, oceans.
        vertical_ratio : float > 0
            How much to exaggerate hills. Kind of arbitrary. 40 is reasonable,
            but try bigger and smaller values!

        Returns
        -------
        np.ndarray
            Processed data.
        """
        if values is None:
            values = self.get_elevation_data()
        nan_vals = np.isnan(values)

        values[nan_vals] = np.nanmin(values)
        self.minv = np.min(values)
        self.maxv = np.max(values)
        self.vertical_ratio = vertical_ratio
        values = (values - self.minv) / (self.maxv - self.minv)

        is_water = values < np.percentile(values, water_ntile)
        is_lake = rank.gradient(img_as_ubyte(values), square(3)) < lake_flatness

        values[nan_vals] = np.nan
        values[np.logical_or(is_water, is_lake)] = np.nan
        values = vertical_ratio * values[-1::-1]  # switch north and south
        if bounds is not None:
            if self.mask is None:
                m = self.build_bound_mask(values, bounds)
                self.mask = m
            else:
                m = self.mask
            values += m
        return values
    
    def approx_elevation(self, lat, long, values):
        return values[int(lat), int(long)]
    
    # pylint: disable=too-many-arguments,too-many-locals
    def plot_map_with_extra_lines(
        self,
        values=None,
        extra_lines=[],#should be a list of 3x? arrays with [[lat], [long], [elev]] (if elev is None, elev is inferred as ground level)
        label="The White\nMountains",
        label_x=0.62,
        label_y=0.15,
        label_verticalalignment="bottom",
        label_size=60,
        line_color="black",
        kind="gradient",
        linewidth=2,
        background_color=(0.9255, 0.9098, 0.9255),
        size_scale=20,
        ax=None,
        experimental_zorder=True,
        no_label=True,
        verbose=False,
        plane_color=None,
        plane_width=None
    ):
        """Plot the map.

        Lots of nobs, and they're all useful to sometimes turn.

        Parameters
        ----------
        values : np.ndarray
            Array of elevations to plot. Defaults to the elevations at the provided
            bounding box.
        label : string
            Label to place on the map. Use an empty string for no label.
        label_x : float in [0, 1]
            Where to position the label horizontally
        label_y : float in [0, 1]
            Where to position the label vertically
        label_verticalalignment: "top" or "bottom"
            Whether the label_x and label_y refer to the top or bottom left corner
            of the label text box
        label_size : int
            fontsize of the label
        line_color : string or callable
            colors for the map. A callable will be fed the scaled index in [0, 1]
        kind : {"gradient" | "elevation"}
            If you provide a colormap to `line_color`, "gradient" colors by the line index, and
            "elevation" colors by the actual elevation along the line.
        linewidth : float
            Width of each line in the map
        background_color : color
            For the background of the map and figure
        scale_size : float
            If you are printing this, make this number bigger.
        ax : matplotlib Axes
            You can pass your own axes!

        Returns
        -------
        matplotlib.Axes
        """
        if plane_color is None:
            plane_color = line_color
        if plane_width is None:
            plane_width = linewidth
        if kind not in {"gradient", "elevation"}:
            raise TypeError("Argument `kind` must be one of 'gradient' or 'elevation'")
        if values is None:
            values = self.preprocess()

        if ax is None:
            ratio = (self.lats[1] - self.lats[0]) / (self.longs[1] - self.longs[0])
            _, ax = plt.subplots(figsize=(size_scale, size_scale * ratio))

        x = np.arange(values.shape[1])
        norm = plt.Normalize(np.nanmin(values), np.nanmax(values))

        lats_norm = lambda _x: -values.shape[0] * (_x - self.lats[1])/(self.lats[1] - self.lats[0])
        longs_norm = lambda _x: values.shape[1] * (_x - self.longs[0])/(self.longs[1] - self.longs[0])
        elevs_norm = lambda _x: self.vertical_ratio * ((_x - self.minv) / (self.maxv - self.minv))
        extra_lines = [np.array([
            i[0],
            i[1],
            np.array([
                (elevs_norm(_i) if _i is not None else self.approx_elevation(
                    lats_norm(i[0][n]),
                    longs_norm(i[1][n]),
                    values
                )) for n, _i in enumerate(i[2])
            ])])
            for i in extra_lines
        ]
        if verbose:
            print(extra_lines)
        for idx, (lats, longs, elevs) in enumerate(extra_lines):
            _lats = lats_norm(lats)
            _longs = longs_norm(longs)
            y_base = -6 * _lats * np.ones_like(elevs)
            _y = elevs + y_base
            _x = _longs
            if callable(plane_color) and kind == "elevation":
                points = np.array([_x, _y]).T.reshape((-1, 1, 2))
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                if experimental_zorder:
                    for n, seg in enumerate(segments):
                        color = plane_color(norm(elevs[n]))
                        l = ax.plot(seg[:, 0], seg[:, 1], "-", c=color, zorder=_lats[n], lw=plane_width)
                        #mplcyberpunk.add_gradient_fill(l[0], alpha_gradientglow=.5, gradient_start="min")
                else:
                    lines = LineCollection(
                        segments, cmap=plane_color, zorder=idx, norm=norm
                    )
                    lines.set_array(elevs)
                    lines.set_linewidth(plane_width)
                    ax.add_collection(lines)
            else:
                if callable(plane_color) and kind == "gradient":
                    color = plane_color(_lats[0] / values.shape[0])
                else:
                    color = plane_color
                if experimental_zorder:
                    points = np.array([_x, _y]).T.reshape((-1, 1, 2))
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    for n, seg in enumerate(segments):
                        if callable(plane_color) and kind == "gradient":
                            color = plane_color(_lats[n] / values.shape[0])
                        l = ax.plot(seg[:, 0], seg[:, 1], "-", c=color, zorder=_lats[n])
                        #mplcyberpunk.add_gradient_fill(l[0], alpha_gradientglow=.5, gradient_start="min")
                else:
                    ax.plot(_x, _y, "-", color=color, zorder=_lats[0], lw=plane_width)
            #ax.fill_between(x, y_base, y, color=background_color, zorder=idx)
        mplcyberpunk.add_gradient_fill(alpha_gradientglow=.5, gradient_start="bottom")
        
        for idx, row in enumerate(values):
            y_base = -6 * idx * np.ones_like(row)
            y = row + y_base
            if callable(line_color) and kind == "elevation":
                points = np.array([x, y]).T.reshape((-1, 1, 2))
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lines = LineCollection(
                    segments, cmap=line_color, zorder=idx + 1, norm=norm
                )
                lines.set_array(row)
                lines.set_linewidth(linewidth)
                ax.add_collection(lines)
            else:
                if callable(line_color) and kind == "gradient":
                    color = line_color(idx / values.shape[0])
                else:
                    color = line_color

                ax.plot(x, y, "-", color=color, zorder=idx, lw=linewidth)
            ax.fill_between(x, y_base - 1500, y, color=background_color, zorder=idx)
            ax.fill_between(np.concatenate([x[:1] - 1000, x, x[-1:] + 1000]), np.concatenate([(y_base - 1500)[:1], (y_base - 1500), (y_base - 1500)[-1:]]), np.concatenate([(y_base - 20)[:1], (y_base - 20), (y_base - 20)[-1:]]), color=background_color, zorder=idx)

        if not no_label:
            ax.text(
                label_x,
                label_y,
                label,
                transform=ax.transAxes,
                fontproperties=self.font,
                size=label_size,
                verticalalignment=label_verticalalignment,
                bbox={"facecolor": background_color, "alpha": 1, "linewidth": 0},
                zorder=len(values) + 10,
            )

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(background_color)
        ax.set_ylim((-1800, -140))#((-1665, -275))
        ax.set_xlim((-15, 285))#((5.199999999999999, 264.8))
        return ax