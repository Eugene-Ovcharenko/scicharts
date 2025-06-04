import os
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Sequence, Literal, Tuple, Union, List, Optional, Dict, Any
from fractions import Fraction
from  matplotlib import collections
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

import warnings
warnings.filterwarnings("ignore")

# ======================================================================================================================
# STYLE FUNCTIONS
# ======================================================================================================================
def make_fig_ax(
    length_key: Literal['1', '3/2', '2', '3'],
    height_key: Literal['1', '3/2', '2', '3'],
    margins_top: Union[float, int] = 8,
    margins_bottom: Union[float, int] = 14,
    margins_left: Union[float, int] = 16,
    margins_right: Union[float, int] = 1
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a matplotlib Figure and Axes with dimensions based on specified keys.

    Args:
        length_key (Literal['1', '3/2', '2', '3']): Key for the figure width.
            Maps to a physical width in millimeters:
            '1' = 56 mm, '3/2' = 84 mm, '2' = 112 mm, '3' = 168 mm.
        height_key (Literal['1', '3/2', '2', '3']): Key for the figure height.
            Uses the same mapping as length_key.
        margins_top (Union[float, int]): Top margin in millimeters. Default is 8 mm.
        margins_bottom (Union[float, int]): Bottom margin in millimeters. Default is 14 mm.
        margins_left (Union[float, int]): Left margin in millimeters. Default is 16 mm.
        margins_right (Union[float, int]): Right margin in millimeters. Default is 1 mm.

    Returns:
        Tuple[plt.Figure, plt.Axes]: A tuple containing the created Figure and Axes objects,
        sized in inches and adjusted for the specified margins.

    Raises:
        ValueError: If length_key or height_key is not one of: '1', '3/2', '2', '3'.
    """
    size_mm_dict = {
        '1': 56,
        '3/2': 84,
        '2': 112,
        '3': 168
    }

    if length_key not in size_mm_dict or height_key not in size_mm_dict:
        raise ValueError("Keys must be one of: '1', '3/2', '2', '3'")

    # Convert selected dimensions from millimeters to inches
    width_in = size_mm_dict[length_key] / 25.4
    height_in = size_mm_dict[height_key] / 25.4

    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=300)

    # Calculate normalized margin values (0–1) for subplots_adjust
    left = margins_left / size_mm_dict[length_key]
    right = 1 - (margins_right / size_mm_dict[length_key])
    bottom = margins_bottom / size_mm_dict[height_key]
    top = 1 - (margins_top / size_mm_dict[height_key])

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    return fig, ax


def set_color_style(
        greyscale: bool = False
) -> Tuple[Sequence[Tuple[float, float, float]], Union[str, Tuple[float, float, float]]]:
    """Set a color palette and primary color based on greyscale preference.

    Args:
        greyscale (bool): If True, generate a greyscale palette and use 'dimgray'
            as the primary color. If False, use Seaborn palettes for contrast and
            blue.

    Returns:
        tuple:
            palette_contrast (Sequence[Tuple[float, float, float]]):
                If greyscale is True, a list of five RGB tuples evenly spaced
                from light grey to dark grey. Otherwise, the Seaborn 'Set1' palette
                (a list of RGB tuples).
            color (str or Tuple[float, float, float]):
                If greyscale is True, the string 'dimgray'. Otherwise, the last
                (darkest) RGB tuple from the Seaborn 'Blues' palette.
    """
    if greyscale:
        # Generate five grey levels from 0.7 to 0.0 (light to dark)
        palette_contrast = [(v, v, v) for v in np.linspace(0.7, 0.0, 5)]
        color = 'dimgray'
    else:
        palette_contrast = sns.color_palette('Set1')
        color = sns.color_palette('Blues')[-1]

    return palette_contrast, color


def apply_font_style(
        ax: plt.Axes
) -> None:
    """Configure the global font settings for a Matplotlib Axes.

    Args:
        ax (plt.Axes): The Axes object on which to apply the font style.
            Although font settings are applied globally via rcParams, passing
            the Axes ensures consistency in function signatures when styling plots.

    Returns:
        None
    """
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12
    })


def _apply_axes_style(
        ax: plt.Axes,
        xlabel: str,
        ylabel: str
) -> None:
    """Apply label formatting and general style to a Matplotlib Axes.

    Args:
        ax (plt.Axes): The Axes object to style.
        xlabel (str): The text for the x-axis label. Will be line-broken
            automatically based on the axis width.
        ylabel (str): The text for the y-axis label. Will be line-broken
            automatically based on the axis height.

    Returns:
        None
    """
    # Determine scaling coefficients for automatic line breaking
    bbox = ax.get_window_extent()
    x_text_scale_coef = int(bbox.width / 20)
    y_text_scale_coef = int(bbox.height / 20)

    ax.set_xlabel(
        _auto_linebreak(xlabel, maxlen=x_text_scale_coef),
        labelpad=0,
        linespacing=0.8
    )
    ax.set_ylabel(
        _auto_linebreak(ylabel, maxlen=y_text_scale_coef),
        labelpad=0,
        linespacing=0.8
    )

    # Autoscale the y-axis based on data limits or other custom logic
    _autoscale_yaxis(ax)

    # Hide top and right spines for a cleaner appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def _load_chart_data(
    file_path: str,
    values_col_idx: int = 1,
    group_col_idx: int = 2,
    subgroup_col_idx: Optional[int] = None,
    groups_order: Optional[List[str]] = None,
    subgroups_order: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Read Sheet 1 (main data) and Sheet 2 (pair-wise p-values) from an Excel
    workbook and perform all integrity checks previously done inline.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned main data table.
    pvals : pandas.DataFrame
        Table of p-values.
    meta : dict
        Auxiliary objects required by downstream plotting code.
    """
    print(f"Read file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # ---------- Sheet 1 (main data) ----------
    try:
        df = pd.read_excel(file_path, sheet_name="Sheet1")
        value_col_name = df.columns[values_col_idx]
        group_col_name = df.columns[group_col_idx]

        if subgroup_col_idx is not None:
            subgroup_col_name = df.columns[subgroup_col_idx]
            subgroup_idx = df[subgroup_col_name]
            subgroups = subgroup_idx.unique().tolist()
        else:
            subgroup_col_name = None
            subgroup_idx = None
            subgroups = None

        df[group_col_name] = df[group_col_name].astype(str)
        groups = sorted(df[group_col_name].unique())
        group_sizes = df.groupby(group_col_name).size()

        if groups_order is not None:
            missing = set(groups) - set(groups_order)
            if missing:
                raise ValueError(
                    "groups_order is missing the following labels "
                    f"detected in the data: {sorted(missing)}"
                )
            groups = groups_order

        if subgroups_order is not None and subgroups is not None:
            missing = set(subgroups) - set(subgroups_order)
            if missing:
                raise ValueError(
                    "subgroups_order is missing the following labels "
                    f"detected in the data: {sorted(missing)}"
                )
            subgroups = subgroups_order

        groups_valid = [g for g in groups if group_sizes[g] >= 5]
        if len(groups_valid) < 2:
            raise ValueError(
                "A minimum of two cohorts with at least 5 observations is required."
            )

        print("Data:\n", df.head(), "\n")

    except Exception as e:
        raise RuntimeError(f"Data loading error - Sheet1: {e}") from e

    # ---------- Sheet 2 (p-values) ----------
    try:
        pvals = pd.read_excel(file_path, sheet_name=1)
        pvals = pvals.dropna()
        for col in pvals.columns:
            if col != "p":
                pvals[col] = pvals[col].astype(str)
        print("p-values:\n", pvals, "\n")

    except Exception as e:
        raise RuntimeError(f"Data loading error - Sheet2: {e}") from e

    meta: Dict[str, Any] = dict(
        value_col_name=value_col_name,
        group_col_name=group_col_name,
        subgroup_col_name=subgroup_col_name,
        subgroup_idx=subgroup_idx,
        groups=groups,
        subgroups=subgroups,
        group_sizes=group_sizes,
    )

    return df, pvals, meta


def _save_chart(
    fig: plt.Figure,
    file_path: str,
    figure_length_key: str,
    figure_height_key: str,
    greyscale: bool,
    dpi: int = 300,
    transparent: bool = False,
) -> None:
    """
    Save a Matplotlib figure to disk using the naming convention previously
    applied inline.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to be saved.
    file_path : str
        Original Excel file path; filename stem is reused for the output.
    figure_length_key, figure_height_key : str
        Keys ("1", "3/2", "2", "3") converted to their numeric ratios for
        inclusion in the output filename.
    greyscale : bool
        Whether the figure is grayscale; controls the filename suffix.
    dpi : int, default 300
        Resolution passed to `fig.savefig`.
    transparent : bool, default False
        Transparency flag passed to `fig.savefig`.
    """
    postfix = f"{float(Fraction(figure_length_key))}x{float(Fraction(figure_height_key))}"
    base, _ = os.path.splitext(file_path)
    suffix = "grays" if greyscale else "color"
    save_file_path = f"{base}_{postfix}_{suffix}.png"

    fig.savefig(save_file_path, dpi=dpi, transparent=transparent)
    plt.close(fig)


def _auto_linebreak(
        title: str,
        maxlen: int = 30
) -> str:
    """
    Automatically insert a line break in the axis title if it's long.
    Splits at the nearest space before maxlen.

    Args:
        title (str): Axis title text.
        maxlen (int): Max chars before breaking.

    Returns:
        str: Title with '\n' for matplotlib.
    """
    if len(title) <= maxlen:
        return title

    space_idx = title.rfind(' ', 0, maxlen)
    if space_idx == -1:
        space_idx = maxlen  # No space found, just cut
    return title[:space_idx] + '\n' + title[space_idx + 1:]


def _draw_significance_bar_abs(
    ax: plt.Axes,
    group_locs: Tuple[float, float],
    p_value: Union[float, int],
    y_mm: float,
    bar_height_mm: float = 1.2,
    alpha: float = 0.05,
    gap_px: int = 2
) -> None:
    """Draw a significance bar between two groups on a Matplotlib Axes.

    Args:
        ax (plt.Axes): The Axes object on which to draw the significance bar.
        group_locs (Tuple[float, float]): X-axis positions of the two groups.
        p_value (Union[float, int]): P-value for the statistical test. Must be a
            numeric type. No bar is drawn if p_value is not numeric or p_value >= alpha.
        y_mm (float): Vertical position for the base of the bar, in millimeters.
        bar_height_mm (float): Height of the significance bar, in millimeters.
            Defaults to 1.2 mm.
        alpha (float): Significance threshold. A bar is drawn only if p_value < alpha.
            Defaults to 0.05.
        gap_px (int): Horizontal gap in pixels to offset the bar from the group locations.
            Defaults to 2 pixels.

    Returns:
        None: The function modifies the Axes in place and does not return a value.
    """
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in = bbox.height
    height_mm = height_in * 25.4

    # Do not draw if p_value is not numeric or not significant
    if not isinstance(p_value, (int, float)) or p_value >= alpha:
        return

    # Determine significance label based on p_value
    if p_value < 0.001:
        label = "***"
    elif p_value < 0.01:
        label = "**"
    else:
        label = "*"

    left, right = group_locs
    if left > right:
        left, right = right, left

    # Convert data coordinates to pixel coordinates, adjust by gap, then convert back
    x_pixel_left = ax.transData.transform((left, 0))[0]
    x_pixel_right = ax.transData.transform((right, 0))[0]
    left_new = ax.transData.inverted().transform((x_pixel_left + gap_px, 0))[0]
    right_new = ax.transData.inverted().transform((x_pixel_right - gap_px, 0))[0]

    # Convert mm-based y position and bar height into fraction of axis height
    y_frac = y_mm / height_mm
    bar_height_frac = bar_height_mm / height_mm

    # Draw the bar
    ax.plot(
        [left_new, left_new, right_new, right_new],
        [y_frac, y_frac + bar_height_frac, y_frac + bar_height_frac, y_frac],
        lw=1.0,
        c='black',
        transform=ax.get_xaxis_transform(),
        clip_on=False
    )

    # Place the significance label above the bar
    ax.annotate(
        label,
        xy=((left_new + right_new) / 2, y_frac - bar_height_frac * 3),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=10,
        color='black',
        clip_on=False
    )


def _draw_significance_bars(
    ax: plt.Axes,
    pvals: pd.DataFrame,
    group_col_name: str,
    subgroup_col_name: Optional[str],
    above_mm: float = 2.0,
    bar_height_mm: float = 1.2,
    mm_step: float = 2.5,
    alpha: float = 0.05,
    box_plot_width_fraction: float = 0.8,
    groups_order: Optional[List[str]] = None,
    subgroups_order: Optional[List[str]] = None,
    y_autoleveling: bool = True
) -> None:
    """Draw multiple significance bars on a Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    pvals : pandas.DataFrame
        DataFrame of pairwise comparisons; must contain column "p".
    group_col_name : str
        Prefix of columns specifying primary groups in `pvals`.
    subgroup_col_name : str or None
        Prefix of columns specifying sub-groups in `pvals`.
    above_mm : float, default 2.0
        Offset (mm) above reference level for the first bar.
    bar_height_mm : float, default 1.2
        Height (mm) of each bar.
    mm_step : float, default 2.5
        Vertical spacing (mm) between stacked bars.
    alpha : float, default 0.05
        Bars are drawn only for *p* < `alpha`.
    box_plot_width_fraction : float, default 0.8
        Fractional width reserved for one primary group on the x-axis.
    groups_order : list[str] or None
        Left-to-right order of primary groups as plotted.
    subgroups_order : list[str] or None
        Order of sub-groups (hue) inside each primary group.
    y_autoleveling : bool, default False
        If True, bars are placed above the maximum y of the compared groups
        instead of the global maximum of the axis.
    """
    # ---------- collect global y coordinates ----------
    all_y_coords: List[float] = []
    for patch in ax.patches:
        if hasattr(patch, "get_y") and hasattr(patch, "get_height"):
            all_y_coords.append(patch.get_y() + patch.get_height())
    for line in ax.get_lines():
        y_data = line.get_ydata()
        if isinstance(y_data, (list, np.ndarray)) and len(y_data) > 0:
            all_y_coords.extend(np.asarray(y_data).flatten())
    for collection in ax.collections:
        if hasattr(collection, "get_offsets"):
            offsets = collection.get_offsets()
            if offsets.ndim == 2 and offsets.shape[1] == 2 and offsets.size > 0:
                all_y_coords.extend(offsets[:, 1])

    y_max_data = np.nanmax(all_y_coords) if all_y_coords else ax.get_ylim()[1]
    y_min_ax, y_max_ax = ax.get_ylim()
    y_frac_top = (y_max_data - y_min_ax) / (y_max_ax - y_min_ax) if (y_max_ax - y_min_ax) else 1.0

    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in = bbox.height
    if height_in == 0:
        return

    height_mm = height_in * 25.4
    mm_first_bar_global = y_frac_top * height_mm + above_mm

    # ---------- filter significant comparisons ----------
    pvals_to_draw = pvals[pvals["p"] < alpha].copy()
    if pvals_to_draw.empty:
        return

    # ---------- resolve plotting order ----------
    group_cols = [c for c in pvals_to_draw.columns if c.startswith(group_col_name)]
    if groups_order is None:
        groups_order = [tick.get_text() for tick in ax.get_xticklabels()]
    groups = list(map(str, groups_order))

    if subgroup_col_name:
        subgroup_cols = [c for c in pvals_to_draw.columns if c.startswith(subgroup_col_name)]
        if subgroups_order is None:
            _, sub_lbls = ax.get_legend_handles_labels()
            subgroups_order = sub_lbls
        subgroups = list(map(str, subgroups_order))
    else:
        subgroups = []
        subgroup_cols = []

    # ---------- per-group y maxima (for local reference) ----------
    group_y_max: Dict[int, float] = {}
    if y_autoleveling:
        # swarm/scatter
        for coll in ax.collections:
            if hasattr(coll, "get_offsets"):
                offs = coll.get_offsets()
                if offs.ndim == 2 and offs.shape[1] == 2:
                    for x_val, y_val in offs:
                        gi = int(round(x_val))
                        group_y_max[gi] = max(group_y_max.get(gi, -np.inf), y_val)
        # boxplot patches
        for patch in ax.patches:
            if all(hasattr(patch, a) for a in ("get_x", "get_width", "get_y", "get_height")):
                x_c = patch.get_x() + patch.get_width() / 2
                gi = int(round(x_c))
                y_v = patch.get_y() + patch.get_height()
                group_y_max[gi] = max(group_y_max.get(gi, -np.inf), y_v)

    # ---------- build list of bars ----------
    bars_to_draw: List[Dict[str, float]] = []
    for _, row in pvals_to_draw.iterrows():
        g1, g2 = row[group_cols[0]], row[group_cols[1]]
        sg1 = row[subgroup_cols[0]] if subgroup_cols else None
        sg2 = row[subgroup_cols[1]] if len(subgroup_cols) > 1 else None
        p_val = row["p"]
        x_coords: List[float] = []

        for g_name, sg_name in [(g1, sg1), (g2, sg2)]:
            g_idx = groups.index(g_name)
            if sg_name is not None and subgroups:
                sub_idx = [str(sg) for sg in subgroups].index(str(sg_name))
                width_slot = box_plot_width_fraction / len(subgroups)
                x_coord = g_idx - box_plot_width_fraction / 2 + width_slot * (sub_idx + 0.5)
            else:
                x_coord = float(g_idx)
            x_coords.append(x_coord)

        if len(x_coords) != 2 or abs(x_coords[0] - x_coords[1]) < 1e-6:
            continue
        bars_to_draw.append({"x1": x_coords[0], "x2": x_coords[1], "width": abs(x_coords[1] - x_coords[0]), "p_val": p_val})

    bars_to_draw.sort(key=lambda d: d["width"])

    # ---------- place bars without overlap ----------
    occupied_at_level: Dict[int, List[Tuple[float, float]]] = {}
    for bar in bars_to_draw:
        x1, x2, p_val = bar["x1"], bar["x2"], bar["p_val"]
        x_range = tuple(sorted((x1, x2)))
        level = 0
        while True:
            if level not in occupied_at_level:
                occupied_at_level[level] = []
            if not any(max(x_range[0], r0) < min(x_range[1], r1) - 1e-3 for r0, r1 in occupied_at_level[level]):
                occupied_at_level[level].append(x_range)
                break
            level += 1
            if level > 20:  # safety limit
                warnings.warn("Exceeded 20 stacked significance levels.")
                occupied_at_level[level - 1].append(x_range)
                level -= 1
                break

        # reference level for this bar
        if y_autoleveling and group_y_max:
            ref_y = max(group_y_max.get(int(round(x1)), y_max_data),
                        group_y_max.get(int(round(x2)), y_max_data))
            y_frac = (ref_y - y_min_ax) / (y_max_ax - y_min_ax) if (y_max_ax - y_min_ax) else 1.0
            mm_first_bar = y_frac * height_mm + above_mm
        else:
            mm_first_bar = mm_first_bar_global

        y_mm = mm_first_bar + level * mm_step
        _draw_significance_bar_abs(
            ax,
            (x1, x2),
            p_val,
            y_mm,
            bar_height_mm=bar_height_mm,
            alpha=alpha
        )


def _autoscale_yaxis(
        ax: plt.Axes,
        min_ticks: int = 4,
        max_ticks: int = 8
) -> None:
    """Automatically scale the Y-axis and set “nice” tick marks.

    This function computes an appropriate Y-axis range starting at zero,
    selects a “nice” tick interval based on the data span, and enforces
    a minimum and maximum number of ticks.

    Args:
        ax (plt.Axes): The Axes object whose Y-axis will be rescaled.
        min_ticks (int): Minimum number of ticks to display on the Y-axis.
            Defaults to 4.
        max_ticks (int): Maximum number of ticks to display on the Y-axis.
            Defaults to 8.

    Returns:
        None: Modifies the Axes in place by setting yticks and disabling
        automatic y-axis scaling.
    """
    # Force lower bound of Y-axis to 0
    ax.set_ylim(0)
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        ymax = ymin + 1

    # Round minimum down (usually 0 for box plots)
    y_min = 0 if ymin < 0.05 * ymax else math.floor(ymin)
    span = ymax - y_min

    # Determine raw step based on desired minimum ticks
    raw_step = span / max(min_ticks, 1)

    # Compute magnitude for “nice” step sizes
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step > 0 else 1
    nice_factors = np.array([1, 2, 2.5, 5, 10])
    step_candidates = nice_factors * magnitude

    # Select a step such that number of ticks is between min_ticks and max_ticks
    for step in step_candidates:
        ticks = np.arange(
            math.floor(y_min / step) * step,
            math.ceil(ymax / step) * step + step * 0.5,
            step
        )
        if min_ticks <= len(ticks) <= max_ticks:
            break
    else:
        # Fallback: divide span into 4 intervals if no “nice” step works
        step = max(raw_step, 1)
        ticks = np.linspace(y_min, ymax, 4)

    # Format tick labels: use two decimals if step < 1, else integer formatting
    if step < 1:
        tick_labels = [f"{tick:.2f}".rstrip("0").rstrip(".") for tick in ticks]
    else:
        tick_labels = [f"{int(tick)}" for tick in ticks]

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_autoscaley_on(False)


def _set_comma_decimal(
    ax: plt.Axes,
    ndigits: int = 3,
    axes: str = "y",            # ← new: "both", "x", or "y"
) -> None:
    """
    Replace the decimal point by a comma on numeric axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ndigits : int
        Digits after the decimal for non-integer ticks.
    axes : {"both", "x", "y"}, default "y"
        Which axis to format.  Use "y" to avoid changing categorical x-labels.
    """

    def _fmt(x: float, _: Any) -> str:
        if math.isclose(x % 1, 0.0, abs_tol=1e-12):
            return f"{int(round(x))}"
        s = f"{x:.{ndigits}f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")

    def _axis_is_numeric(axis) -> bool:
        axis.figure.canvas.draw_idle()           # ensure labels exist
        labels = [t.get_text() for t in axis.get_ticklabels()]
        locs   = axis.get_ticklocs()

        for lbl, loc in zip(labels, locs):
            if not lbl:
                continue
            try:
                val = float(lbl.replace(",", "."))
            except ValueError:
                return False                     # non-numeric label → categorical
            if not math.isclose(val, loc):       # mismatch text vs. coordinate
                return False
        return True

    formatter = FuncFormatter(_fmt)

    if axes in ("both", "x") and _axis_is_numeric(ax.xaxis):
        ax.xaxis.set_major_formatter(formatter)
    if axes in ("both", "y") and _axis_is_numeric(ax.yaxis):
        ax.yaxis.set_major_formatter(formatter)


# ======================================================================================================================
# PLOTTING FUNCTIONS
# ======================================================================================================================
def boxplot_builder(
    file_path: str,
    values_col_idx: int = 1,
    group_col_idx: int = 2,
    subgroup_col_idx: Optional[int] = None,
    figure_length_key: Literal['1', '3/2', '2', '3'] = '1',
    figure_height_key: Literal['1', '3/2', '2', '3'] = '1',
    num_format_ru: bool = True,
    show_outliers: bool = True,
    groups_order: Optional[List[str]] = None,
    subgroups_order: Optional[List[str]] = None,
) -> None:
    """
    Build and save boxplots (colour and grayscale) from Excel data, with
    optional swarm overlays and significance bars.  The plotting order of
    groups and sub-groups can now be specified explicitly.

    Parameters
    ----------
    file_path : str
        Path to the Excel file containing chart data (Sheet1) and p-values
        (Sheet2).
    values_col_idx : int, default 1
        Zero-based index of the numeric column in Sheet1.
    group_col_idx : int, default 2
        Zero-based index of the primary-group column in Sheet1.
    subgroup_col_idx : int or None, default None
        Zero-based index of the subgroup column.  If None, no subgrouping.
    figure_length_key, figure_height_key : {'1','3/2','2','3'}, default '1'
        Keys mapped to journal-style figure dimensions in millimetres
        (56, 84, 112, 168 mm).
    num_format_ru : bool, default True
        If True, axis tick labels use a comma as the decimal mark.
    show_outliers : bool, keyword-only, default True
        Controls whether statistical outliers (fliers) are drawn in the boxplot.
    groups_order : list[str] or None, keyword-only
        Desired left-to-right order of primary groups.  Must include *all*
        group labels present in the data.
    subgroups_order : list[str] or None, keyword-only
        Desired hue order of sub-groups inside every primary group.
        Must include *all* sub-group labels present in the data.
    """
    # Apply global font settings to current Axes
    apply_font_style(plt.gca())

    # Read file
    df, pvals, meta = _load_chart_data(
        file_path=file_path,
        values_col_idx=values_col_idx,
        group_col_idx=group_col_idx,
        subgroup_col_idx=subgroup_col_idx,
        groups_order=groups_order,
        subgroups_order=subgroups_order,
    )
    value_col_name = meta["value_col_name"]
    group_col_name = meta["group_col_name"]
    subgroup_col_name = meta["subgroup_col_name"]
    subgroup_idx = meta["subgroup_idx"]
    groups = meta["groups"]
    subgroups = meta["subgroups"]
    group_sizes = meta["group_sizes"]

    # Iterate twice: once for color, once for grayscale
    for greyscale in [False, True]:
        palette_contrast, color = set_color_style(greyscale=greyscale)
        basic_color_palette = [color] * len(groups)
        palette = palette_contrast if (subgroup_col_idx is not None) else basic_color_palette

        # Create figure and axes with specified dimensions
        fig, ax = make_fig_ax(
            length_key=figure_length_key,
            height_key=figure_height_key
        )

        # Prepare data for boxplot: set values to NaN for small groups
        df_box = df.copy()
        for g, size in group_sizes.items():
            if size < 5:
                df_box.loc[df_box[group_col_name] == g, value_col_name] = np.nan

        # Draw boxplot (transparent fill, black edges)
        sns.boxplot(
            data=df_box,
            x=group_col_name,
            y=value_col_name,
            ax=ax,
            hue=subgroup_idx,
            order=groups,
            hue_order=subgroups,
            whis= 1.5 if show_outliers else 9,
            gap = 0.1,
            boxprops=dict(facecolor='none', edgecolor='black'),
            whiskerprops=dict(color='black', linestyle='-', linewidth=1.25),
            capprops=dict(color='black'),
            medianprops=dict(color='black'),
            flierprops=dict(
                marker='+',
                markerfacecolor='black',
                markeredgecolor='black',
                markersize=3
            ),
            showfliers=True,
            legend=False,
            color='white',
            zorder=1
        )

        # Draw swarmplot (dots) over boxplot
        if subgroup_col_idx is not None:
            flag_dodge = True
        else:
            flag_dodge = False
        swarm = sns.swarmplot(
            data=df,
            x=df.columns[group_col_idx],
            y=df.columns[values_col_idx],
            hue=subgroup_idx,
            order=groups,
            hue_order=subgroups,
            marker='o',
            dodge=flag_dodge,
            palette=palette,
            legend=True,
            size=3,
            alpha=0.7,
            ax=ax,
            zorder=0
        )

        # Configure x-axis ticks and labels
        n_groups = len(groups)
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlim(-0.5, n_groups - 0.5)

        # Apply axis label formatting and styling
        _apply_axes_style(
            ax=ax,
            xlabel=group_col_name,
            ylabel=value_col_name
        )

        # Format numeric tick labels with Russian-style decimals if requested
        if num_format_ru:
            _set_comma_decimal(ax)

        # Remove any existing legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            for lg in fig.legends:
                lg.remove()
            fig.legends.clear()

        # Rebuild legend from swarmplot handles and labels
        handles, labels = swarm.get_legend_handles_labels()
        if "" in labels:
            handles = handles[1:]
            labels = labels[1:]
        if handles and labels and len(handles) == len(labels):
            fig.legend(
                handles,
                labels,
                loc='lower left',
                bbox_to_anchor=(0, 0),
                ncol=len(labels),
                frameon=False,
                fontsize=10,
                markerscale=1.5,
                handlelength=0.5,
                handletextpad=0.1,
                borderaxespad=0,
                borderpad=0,
                columnspacing=0.1
            )

        # Draw all significance bars for p-values < alpha
        _draw_significance_bars(
            ax=ax,
            pvals=pvals,
            group_col_name=group_col_name,
            subgroup_col_name=subgroup_col_name,
            groups_order=groups,
            subgroups_order=subgroups
        )

        # save figure
        _save_chart(
            fig=fig,
            file_path=file_path,
            figure_length_key=figure_length_key,
            figure_height_key=figure_height_key,
            greyscale=greyscale,
        )

    print("-" * 50)


def barplot_builder(
    file_path: str,
    group_col_idx: int = 1,
    subgroup_col_idx: Optional[int] = None,
    figure_length_key: Literal['1', '3/2', '2', '3'] = '1',
    figure_height_key: Literal['1', '3/2', '2', '3'] = '1',
    num_format_ru: bool = True,
    groups_order: Optional[List[str]] = None,
    subgroups_order: Optional[List[str]] = None,
) -> None:
    """Build and save barplots (color and grayscale) from Excel data with significance bars.

    Reads grouping and (optionally) subgrouping information from an Excel file,
    constructs a side‐by‐side count bar chart for each group (and subgroup),
    overlays statistical significance bars for pairwise comparisons, and saves
    the resulting figures in both color and grayscale variants.

    Args:
        file_path (str):
            Path to the Excel file containing chart data on the first sheet
            and pairwise p-values on the second sheet.
        group_col_idx (int, optional):
            Zero-based index of the column in Sheet1 that defines the primary
            grouping. Defaults to 1.
        subgroup_col_idx (Optional[int], optional):
            Zero-based index of the column in Sheet1 that defines subgroups
            within each primary group. If None, no subgrouping is applied.
            Defaults to None.
        figure_length_key (Literal['1', '3/2', '2', '3'], optional):
            Key for the figure width, mapping to a physical width in millimeters:
            '1'→56 mm, '3/2'→84 mm, '2'→112 mm, '3'→168 mm. Defaults to '1'.
        figure_height_key (Literal['1', '3/2', '2', '3'], optional):
            Key for the figure height, with the same mapping as
            `figure_length_key`. Defaults to '1'.
        num_format_ru (bool, optional):
            If True, tick labels on both axes are formatted with a comma as the
            decimal separator. Defaults to True.
        groups_order (Optional[List[str]], optional):
            Explicit left-to-right order of primary group labels. Must include
            all group names present in the data. If None, the sorted order of
            unique group labels is used. Defaults to None.
        subgroups_order (Optional[List[str]], optional):
            Explicit ordering of subgroup labels (hue order) within each primary
            group. Must include all subgroup labels present in the data. If None,
            the native order of subgroups in the dataset is used. Defaults to None.

    Returns:
        None
    """
    # Apply global font settings to current Axes
    apply_font_style(plt.gca())

    # Read file
    df, pvals, meta = _load_chart_data(
        file_path=file_path,
        group_col_idx=group_col_idx,
        subgroup_col_idx=subgroup_col_idx,
        groups_order=groups_order,
        subgroups_order=subgroups_order,
    )
    group_col_name = meta["group_col_name"]
    subgroup_col_name = meta["subgroup_col_name"]
    subgroup_idx = meta["subgroup_idx"]
    groups = meta["groups"]
    subgroups = meta["subgroups"]
    group_sizes = meta["group_sizes"]

    # Iterate twice: once for color, once for grayscale
    for greyscale in [False, True]:
        palette_contrast, color = set_color_style(greyscale=greyscale)
        basic_color_palette = [color] * len(groups)
        palette = palette_contrast if (subgroup_col_idx is not None) else basic_color_palette

        # Create figure and axes with specified dimensions
        fig, ax = make_fig_ax(
            length_key=figure_length_key,
            height_key=figure_height_key
        )

        ## Prepare pivot table for barchart
        df_plot = df.copy()
        pivot_tab = pd.crosstab(df_plot[group_col_name], df_plot[subgroup_col_name])
        pivot_tab = pivot_tab.reset_index().melt(id_vars=group_col_name, var_name=subgroup_col_name, value_name='count')

        bar_chart = sns.barplot(
            data=pivot_tab,
            x=group_col_name,
            y='count',
            hue=subgroup_col_name,
            hue_order=subgroups,
            ax=ax,
            palette=palette,
            edgecolor='black',
            linewidth=1,
            alpha=0.75,
            width=0.5,
            zorder=1
        )

        # Configure x-axis ticks and labels
        n_groups = len(groups)
        ax.set_xticks(np.arange(n_groups))
        ax.set_xticklabels([str(g) for g in groups])
        ax.set_xlim(-0.5, n_groups - 0.5)

        # Apply axis label formatting and styling
        _apply_axes_style(
            ax=ax,
            xlabel=group_col_name,
            ylabel='Количество' if num_format_ru else 'Count'
        )

        # Format numeric tick labels with Russian-style decimals if requested
        if num_format_ru:
            _set_comma_decimal(ax)

        # Remove any existing legends
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            for lg in fig.legends:
                lg.remove()
            fig.legends.clear()

        # Rebuild legend from swarmplot handles and labels
        handles, labels = bar_chart.get_legend_handles_labels()
        if "" in labels:
            handles = handles[1:]
            labels = labels[1:]
        if handles and labels and len(handles) == len(labels):
            fig.legend(
                handles,
                labels,
                loc='lower left',
                bbox_to_anchor=(0, 0),
                ncol=len(labels),
                frameon=False,
                fontsize=10,
                markerscale=1.5,
                handlelength=1,
                handletextpad=0.1,
                borderaxespad=0.0,
                borderpad=0.0,
                columnspacing=0.5
            )

        # Draw all significance bars for p-values < alpha
        _draw_significance_bars(
            ax=ax,
            pvals=pvals,
            group_col_name=group_col_name,
            subgroup_col_name=subgroup_col_name,
            groups_order=groups,
            subgroups_order=subgroups
        )

        # save figure
        _save_chart(
            fig=fig,
            file_path=file_path,
            figure_length_key=figure_length_key,
            figure_height_key=figure_height_key,
            greyscale=greyscale,
        )

    print("-" * 50)


# ======================================================================================================================
# MAIN FUNCTION
# ======================================================================================================================
def main():
    file_dir = 'charts/svd_clustering'

    file_name = 'fig1_ru.xlsx'
    boxplot_builder(
        file_path=os.path.join(file_dir, file_name),
        values_col_idx=1,
        group_col_idx = 2,
        subgroup_col_idx=3,
        figure_length_key='3/2',
        figure_height_key='3/2',
        num_format_ru=True,
        show_outliers=False,
        groups_order = ['АК', 'МК', 'ТК'],
        subgroups_order = ['ПЭ', 'СКД'],
    )

    # TODO: #2 circle or bar
    file_name = 'fig2_ru.xlsx'

    # TODO: #5 boxplots
    file_names = ['fig5A_ru.xlsx', 'fig5B_ru.xlsx']

    # TODO: #8 scatter + confidence interval
    file_names = ['fig8A_ru.xlsx', 'fig8B_ru.xlsx', 'fig8C_ru.xlsx']

    # TODO: #9 bar
    file_name = 'fig9_ru.xlsx'

    # TODO: #10A-D bar
    file_names = ['fig10A_ru.xlsx', 'fig10B_ru.xlsx', 'fig10C_ru.xlsx', 'fig10D_ru.xlsx']
    file_name = 'fig10A_ru.xlsx'
    barplot_builder(
        file_path=os.path.join(file_dir, file_name),
        group_col_idx=1,
        subgroup_col_idx=2,
        figure_length_key='1',
        figure_height_key='1',
        num_format_ru=True,
        groups_order = None,
        subgroups_order = None,
    )


    # TODO: #10E-G boxplots | show_outliers=True,
    file_names = ['fig10E_ru.xlsx', 'fig10F_ru.xlsx', 'fig10G_ru.xlsx']


if __name__ == '__main__':
    main()

