import os
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns

from typing import Sequence, Literal, Tuple, Union, List, Optional, Dict, Any
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


def apply_axes_style(
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
        auto_linebreak(xlabel, maxlen=x_text_scale_coef),
        labelpad=0,
        linespacing=0.8
    )
    ax.set_ylabel(
        auto_linebreak(ylabel, maxlen=y_text_scale_coef),
        labelpad=0,
        linespacing=0.8
    )

    # Autoscale the y-axis based on data limits or other custom logic
    autoscale_yaxis(ax)

    # Hide top and right spines for a cleaner appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def auto_linebreak(
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


def draw_significance_bar_abs(
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


def draw_significance_bars(
    ax: plt.Axes,
    pvals: pd.DataFrame,
    group_col_name: str,
    subgroup_col_name: Optional[str],
    above_mm: float = 2.0,
    bar_height_mm: float = 1.2,
    mm_step: float = 2.5,
    alpha: float = 0.05,
    box_plot_width_fraction: float = 0.8
) -> None:
    """Draw multiple significance bars on a Matplotlib Axes.

    For each pairwise comparison in the `pvals` DataFrame where the p-value is
    below the `alpha` threshold, this function computes the x-axis locations
    (optionally accounting for subgroups), determines appropriate vertical
    placement to avoid overlaps, and draws a bar with the corresponding
    significance annotation.

    Args:
        ax (plt.Axes): Axes object on which to draw significance bars.
        pvals (pd.DataFrame): DataFrame containing pairwise comparison results.
            Must include a column "p" for p-values and columns starting with
            `group_col_name` (and, if applicable, `subgroup_col_name`) for
            identifying groups in each comparison.
        group_col_name (str): Prefix of columns in `pvals` that identify the
            primary groups for each comparison. Exactly two such columns must
            exist for each row.
        subgroup_col_name (Optional[str]): Prefix of columns in `pvals` that
            identify subgroups within each primary group. If None, subgroups
            are ignored. If provided, exactly two such columns must exist per
            row.
        above_mm (float): Vertical offset in millimeters above the highest data
            point to place the first significance bar. Defaults to 2.0 mm.
        bar_height_mm (float): Height of each significance bar in millimeters.
            Defaults to 1.2 mm.
        mm_step (float): Vertical spacing in millimeters between stacked bars.
            Defaults to 2.5 mm.
        alpha (float): Significance threshold. Only comparisons with p < alpha
            are drawn. Defaults to 0.05.
        box_plot_width_fraction (float): Fractional width allocated for each
            primary group in the box plot. Used to offset subgroups horizontally.
            Defaults to 0.8.

    Returns:
        None: Modifies the `ax` in place to add significance bars.
    """
    # Collect all y-coordinates from bars, lines, and scatter points to find the top
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

    y_max_data = (
        np.nanmax(all_y_coords) if all_y_coords else ax.get_ylim()[1]
    )
    y_min_ax, y_max_ax = ax.get_ylim()
    if (y_max_ax - y_min_ax) != 0:
        y_frac_top = (y_max_data - y_min_ax) / (y_max_ax - y_min_ax)
    else:
        y_frac_top = 1.0

    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in = bbox.height
    if height_in == 0:
        return

    height_mm = height_in * 25.4
    mm_at_top_data = y_frac_top * height_mm
    mm_first_bar = mm_at_top_data + above_mm

    # Filter p-values below alpha
    pvals_to_draw = pvals[pvals["p"] < alpha].copy()
    if pvals_to_draw.empty:
        return

    # Identify group and subgroup names
    group_cols = [
        col for col in pvals_to_draw.columns if col.startswith(group_col_name)
    ]
    groups = pvals_to_draw[group_cols].stack().unique().tolist()

    if subgroup_col_name:
        subgroup_cols = [
            col for col in pvals_to_draw.columns
            if col.startswith(subgroup_col_name)
        ]
        subgroups = pvals_to_draw[subgroup_cols].stack().unique().tolist()
    else:
        subgroups = []

    bars_to_draw: List[Dict[str, float]] = []
    for _, row in pvals_to_draw.iterrows():
        group_name_1 = row.loc[group_cols[0]]
        group_name_2 = (
            row.loc[group_cols[1]] if len(group_cols) > 1 else None
        )

        if subgroup_col_name:
            subgroup_name_1 = row.loc[subgroup_cols[0]]
            subgroup_name_2 = (
                row.loc[subgroup_cols[1]] if len(subgroup_cols) > 1 else None
            )
        else:
            subgroup_name_1 = None
            subgroup_name_2 = None

        p_val = row["p"]
        current_x_coords: List[float] = []

        # Determine x-coordinates for each group (and subgroup if present)
        for group_name, subgroup_name in [
            (group_name_1, subgroup_name_1),
            (group_name_2, subgroup_name_2)
        ]:
            group_idx = groups.index(group_name)
            if subgroup_name is not None and subgroups:
                try:
                    subgroups_str_list = [str(sg) for sg in subgroups]
                    subgroup_idx = subgroups_str_list.index(str(subgroup_name))
                    num_sg_total = len(subgroups_str_list)
                    if num_sg_total == 1:
                        x_coord = float(group_idx)
                    else:
                        width_per_slot = box_plot_width_fraction / num_sg_total
                        x_coord = (
                            float(group_idx)
                            - box_plot_width_fraction / 2
                            + width_per_slot * (subgroup_idx + 0.5)
                        )
                    current_x_coords.append(x_coord)
                except ValueError:
                    warnings.warn(
                        f"Subgroup '{subgroup_name}' not found in {subgroups_str_list}. "
                        f"Using center of group '{group_name}'. Row: {row.to_dict()}"
                    )
                    current_x_coords.append(float(group_idx))
            else:
                current_x_coords.append(float(group_idx))

        if None in current_x_coords or len(current_x_coords) != 2:
            continue

        x1, x2 = current_x_coords
        if abs(x1 - x2) < 1e-6:
            warnings.warn(
                f"Zero-width significance line (x1={x1}, x2={x2}). Skipping. "
                f"Row: {row.to_dict()}"
            )
            continue

        width = abs(x2 - x1)
        bars_to_draw.append({"x1": x1, "x2": x2, "width": width, "p_val": p_val})

    # Sort bars so shorter spans are drawn closer to the data
    bars_to_draw.sort(key=lambda d: d["width"], reverse=False)

    # Place bars on successive vertical levels to avoid overlap
    occupied_ranges_at_level: Dict[int, List[Tuple[float, float]]] = {}
    for bar in bars_to_draw:
        x1 = bar["x1"]
        x2 = bar["x2"]
        p_val = bar["p_val"]
        bar_range = tuple(sorted((x1, x2)))
        chosen_level = 0

        while True:
            if chosen_level not in occupied_ranges_at_level:
                occupied_ranges_at_level[chosen_level] = []

            has_overlap = False
            epsilon = 1e-3
            for r_min, r_max in occupied_ranges_at_level[chosen_level]:
                if max(bar_range[0], r_min) < min(bar_range[1], r_max) - epsilon:
                    has_overlap = True
                    break

            if not has_overlap:
                occupied_ranges_at_level[chosen_level].append(bar_range)
                break

            chosen_level += 1
            if chosen_level > 20:
                warnings.warn(
                    "Exceeded maximum levels (20) for significance lines. "
                    "Assigning to last allowed level."
                )
                occupied_ranges_at_level.setdefault(chosen_level - 1, []).append(
                    bar_range
                )
                break

        y_mm_for_bar = mm_first_bar + chosen_level * mm_step
        draw_significance_bar_abs(
            ax,
            (x1, x2),
            p_val,
            y_mm_for_bar,
            bar_height_mm=bar_height_mm,
            alpha=alpha
        )


def autoscale_yaxis(
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


def set_comma_decimal(
        ax: plt.Axes,
        ndigits: int = 3
) -> None:
    """Format tick labels on both axes to use commas as decimal separators.

    This function applies a custom formatter to the x- and y-axis major ticks:
    - Integers are displayed without a decimal separator.
    - Floating-point numbers are rounded to `ndigits` decimal places, trailing zeros
      and the decimal point are removed if not needed, and the period is replaced by a comma.

    Args:
        ax (plt.Axes): The Axes object whose tick labels will be reformatted.
        ndigits (int): Number of decimal places to round to for fractional values.
            Defaults to 3.

    Returns:
        None: Modifies the Axes in place by setting a new major tick formatter.
    """
    def _fmt(x: float, _: Any) -> str:
        # If x is effectively an integer, display without decimal part
        if math.isclose(x % 1, 0.0, abs_tol=1e-12):
            return f"{int(round(x))}"
        # Otherwise, format with `ndigits` decimal places, strip trailing zeros and dot,
        # then replace the decimal point with a comma
        s = f"{x:.{ndigits}f}".rstrip("0").rstrip(".")
        return s.replace(".", ",")

    formatter = FuncFormatter(_fmt)
    ax.xaxis.set_major_formatter(formatter)
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
    num_format_ru: bool = True
) -> None:
    """Build and save boxplots (color and grayscale) from Excel data, with significance bars.

    This function reads data and p-values from an Excel file (Sheet1 and Sheet2),
    constructs boxplots with overlaid swarmplots by group (and optional subgroup),
    applies styling (fonts, axes, number formatting), draws significance bars for
    pairwise comparisons below the alpha threshold, and saves two PNG outputs
    (one color and one grayscale).

    Args:
        file_path (str): Path to the Excel file containing chart data (Sheet1) and
            p-values (Sheet2).
        values_col_idx (int): Zero-based index of the column in Sheet1 containing
            numeric values to plot. Defaults to 1.
        group_col_idx (int): Zero-based index of the column in Sheet1 containing
            group labels. Defaults to 2.
        subgroup_col_idx (Optional[int]): Zero-based index of the column in Sheet1
            containing subgroup labels within each primary group. If None, no
            subgrouping is applied. Defaults to None.
        figure_length_key (Literal['1','3/2','2','3']): Key for figure width (in mm):
            '1' = 56 mm, '3/2' = 84 mm, '2' = 112 mm, '3' = 168 mm. Defaults to '1'.
        figure_height_key (Literal['1','3/2','2','3']): Key for figure height (in mm),
            using the same mapping as figure_length_key. Defaults to '1'.
        num_format_ru (bool): If True, format axis tick labels with commas as decimal
            separators. Defaults to True.

    Returns:
        None: Saves two PNG files (color and grayscale) derived from the input file.

    Raises:
        FileNotFoundError: If the specified Excel file does not exist.
        ValueError: If there are fewer than two groups with at least five observations.
    """
    # Apply global font settings to current Axes
    apply_font_style(plt.gca())

    # Load main data (Sheet1)
    print(f"Read file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        value_col_name = df.columns[values_col_idx]
        group_col_name = df.columns[group_col_idx]

        if subgroup_col_idx is not None:
            subgroup_col_name = df.columns[subgroup_col_idx]
            subgroup_idx = df[subgroup_col_name]
            subgroups = subgroup_idx.unique()
            stripplot_dodge = True
        else:
            subgroup_col_name = None
            subgroup_idx = None
            subgroups = None
            stripplot_dodge = False

        df[group_col_name] = df[group_col_name].astype(str)
        groups = sorted(df[group_col_name].unique())
        group_sizes = df.groupby(group_col_name).size()

        # Filter out groups with fewer than 5 observations
        groups_valid = [g for g in groups if group_sizes[g] >= 5]
        if len(groups_valid) < 2:
            raise ValueError(
                "A minimum of two cohorts with at least 5 observations is required."
            )

        print("Data:\n", df.head(), "\n")
    except Exception as e:
        print(f"Data loading error - Sheet1: {e}")
        return

    # Load p-values (Sheet2)
    try:
        pvals = pd.read_excel(file_path, sheet_name=1)
        pvals = pvals.dropna()
        for col in pvals.columns:
            if col != 'p':
                pvals[col] = pvals[col].astype(str)
        print("p-values:", pvals, "\n")
    except Exception as e:
        print(f"Data loading error - Sheet2: {e}")
        return

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
        swarm = sns.swarmplot(
            data=df,
            x=df.columns[group_col_idx],
            y=df.columns[values_col_idx],
            hue=subgroup_idx,
            order=groups,
            hue_order=subgroups,
            marker='o',
            dodge=stripplot_dodge,
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
        apply_axes_style(
            ax=ax,
            xlabel=group_col_name,
            ylabel=value_col_name
        )

        # Format numeric tick labels with Russian-style decimals if requested
        if num_format_ru:
            set_comma_decimal(ax)

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
        draw_significance_bars(
            ax,
            pvals,
            group_col_name,
            subgroup_col_name
        )

        # Determine output file path (append _grays or _color)
        base, _ = os.path.splitext(file_path)
        if greyscale:
            save_file_path = f"{base}_grays.png"
        else:
            save_file_path = f"{base}_color.png"

        fig.savefig(save_file_path, dpi=300, transparent=False)
        plt.close(fig)

    print("-" * 50)


# ======================================================================================================================
# MAIN FUNCTION
# ======================================================================================================================
def main():
    file_path = 'charts/svd_clusters/test.xlsx'
    boxplot_builder(file_path)
    file_path = 'charts/svd_clusters/test2.xlsx'
    boxplot_builder(file_path, subgroup_col_idx=3)


if __name__ == '__main__':
    main()

