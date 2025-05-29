import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu
from typing import Literal, Tuple, Union, List, Optional
from matplotlib.ticker import MaxNLocator
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
    """
    Create matplotlib fig, ax with width and height in inches
    based on specified keys.

    Args:
        length_key (str): One of '1', '3/2', '2', '3' for width (mm).
        height_key (str): One of '1', '3/2', '2', '3' for height (mm).

    Returns:
        (fig, ax): Matplotlib Figure and Axes.
    """
    size_mm_dict = {
        '1': 56,
        '3/2': 84,
        '2': 112,
        '3': 168
    }
    if length_key not in size_mm_dict or height_key not in size_mm_dict:
        raise ValueError("Keys must be one of: '1', '3/2', '2', '3'")

    width_in = size_mm_dict[length_key] / 25.4
    height_in = size_mm_dict[height_key] / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=300)

    # provide margins
    left = margins_left / size_mm_dict[length_key]
    right = 1 - (margins_right / size_mm_dict[length_key])
    bottom = margins_bottom / size_mm_dict[height_key]
    top = 1 - (margins_top / size_mm_dict[height_key])

    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    return fig, ax


def apply_font_style(ax):
    plt.style.use('default')
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12
    })


def apply_axes_style(ax, xlabel, ylabel):
    bbox = ax.get_window_extent()
    x_text_scale_coef = int(bbox.width/20)
    y_text_scale_coef = int(bbox.height/20)
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

    # axes general style
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def auto_linebreak(title: str, maxlen: int = 30) -> str:
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
    # Find nearest space before maxlen
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
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in = bbox.height
    height_mm = height_in * 25.4

    if not isinstance(p_value, (int, float)) or p_value >= alpha:
        return
    if p_value < 0.001:
        label = "***"
    elif p_value < 0.01:
        label = "**"
    else:
        label = "*"
    left, right = group_locs
    if left > right:
        left, right = right, left

    # Преобразуем координаты в пиксели, добавляем/вычитаем зазор, переводим обратно
    x_pixel_left = ax.transData.transform((left, 0))[0]
    x_pixel_right = ax.transData.transform((right, 0))[0]
    left_new = ax.transData.inverted().transform((x_pixel_left + gap_px, 0))[0]
    right_new = ax.transData.inverted().transform((x_pixel_right - gap_px, 0))[0]

    y_frac = y_mm / height_mm
    bar_height_frac = bar_height_mm / height_mm

    ax.plot(
        [left_new, left_new, right_new, right_new],
        [y_frac, y_frac + bar_height_frac, y_frac + bar_height_frac, y_frac],
        lw=1.0, c='black',
        transform=ax.get_xaxis_transform(),
        clip_on=False
    )
    ax.annotate(
        label,
        xy=((left_new + right_new) / 2, y_frac - bar_height_frac * 3),
        xycoords=ax.get_xaxis_transform(),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center', va='bottom',
        fontsize=10, color='black',
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
):

    # Вспомогательные: вычисление позиций на оси Y (максимальное значение графика)
    all_y_coords = []
    for patch in ax.patches:
        if hasattr(patch, "get_y") and hasattr(patch, "get_height"):
            all_y_coords.append(patch.get_y() + patch.get_height())
    for line in ax.get_lines():
        y_data = line.get_ydata()
        if isinstance(y_data, (list, np.ndarray)) and len(y_data) > 0:
            all_y_coords.extend(np.asarray(y_data).flatten())
    for collection in ax.collections:
        if hasattr(collection, 'get_offsets'):
            offsets = collection.get_offsets()
            if offsets.ndim == 2 and offsets.shape[1] == 2 and offsets.size > 0:
                all_y_coords.extend(offsets[:, 1])
    y_max_data = np.nanmax(all_y_coords) if all_y_coords else ax.get_ylim()[1]
    y_min_ax, y_max_ax = ax.get_ylim()
    y_frac_top = (y_max_data - y_min_ax) / (y_max_ax - y_min_ax) if (y_max_ax - y_min_ax) != 0 else 1.0
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height_in = bbox.height
    if height_in == 0:
        return
    height_mm = height_in * 25.4
    mm_at_top_data = y_frac_top * height_mm
    mm_first_bar = mm_at_top_data + above_mm

    # опредлеление значимых pvals < alpha
    pvals_to_draw = pvals[pvals['p'] < alpha].copy()
    if pvals_to_draw.empty:
        return

    # Определение групп и подгрупп в pvals_to_draw
    group_cols = [col for col in pvals_to_draw.columns if col.startswith(group_col_name)]
    groups = pvals_to_draw[group_cols].stack().unique().tolist()

    if subgroup_col_name:
        subgroup_cols = [col for col in pvals_to_draw.columns if col.startswith(subgroup_col_name)]
        subgroups = pvals_to_draw[subgroup_cols].stack().unique().tolist()

    bars_to_draw = []
    for _, row in pvals_to_draw.iterrows():

        # Извлекаем группы
        group_name_1, group_name_2 = row.loc[group_cols[0]], row.loc[group_cols[1]] if len(group_cols) > 1 else (None, None)
        if subgroup_col_name:
            subgroup_name_1, subgroup_name_2 = row.loc[subgroup_cols[0]], row.loc[subgroup_cols[1]] if len(subgroup_cols) > 1 else (None, None)
        else:
            subgroup_name_1, subgroup_name_2 = None, None
        p_val = row['p']

        # Определение X координат
        current_x_coords = []
        for group_name, subgroup_name in [(group_name_1, subgroup_name_1), (group_name_2, subgroup_name_2)]:
            group_idx = groups.index(group_name)
            if subgroup_name is not None and subgroups is not None and len(subgroups) > 0:
                try:
                    subgroups_str_list = [str(sg) for sg in subgroups]
                    subgroup_idx = subgroups_str_list.index(str(subgroup_name))
                    num_sg_total = len(subgroups_str_list)
                    if num_sg_total == 1:
                        x_coord = float(group_idx)
                    else:
                        width_per_subgroup_slot = box_plot_width_fraction / num_sg_total
                        x_coord = (float(group_idx) - box_plot_width_fraction / 2 +
                                   width_per_subgroup_slot * (subgroup_idx + 0.5))
                    current_x_coords.append(x_coord)
                except ValueError:
                    warnings.warn(f"Подгруппа '{subgroup_name}' не найдена в subgroups: {subgroups_str_list}. Для '{group_name}' используется центр основной группы. Строка p-value: {row.to_dict()}")
                    current_x_coords.append(float(group_idx))
            else:
                current_x_coords.append(float(group_idx))
        if None in current_x_coords or len(current_x_coords) != 2:
            continue
        x1, x2 = current_x_coords[0], current_x_coords[1]
        if abs(x1 - x2) < 1e-6:
            warnings.warn(f"Линия значимости имеет нулевую ширину (x1={x1}, x2={x2}). Пропуск. Строка p-value: {row.to_dict()}")
            continue
        width = abs(x2 - x1)
        bars_to_draw.append({
            "x1": x1,
            "x2": x2,
            "width": width,
            "p_val": p_val
        })

    # Сортировка баров: длинные — выше, короткие — ниже
    bars_to_draw.sort(key=lambda d: d["width"], reverse=False)

    # Размещение баров по уровням и отрисовка
    occupied_ranges_at_level: dict[int, List[Tuple[float, float]]] = {}
    for bar in bars_to_draw:
        x1, x2, p_val = bar["x1"], bar["x2"], bar["p_val"]
        bar_x_range = tuple(sorted((x1, x2)))
        chosen_level = 0
        while True:
            if chosen_level not in occupied_ranges_at_level:
                occupied_ranges_at_level[chosen_level] = []
            has_overlap = False
            epsilon = 1e-3
            for r_min, r_max in occupied_ranges_at_level[chosen_level]:
                if max(bar_x_range[0], r_min) < min(bar_x_range[1], r_max) - epsilon:
                    has_overlap = True
                    break
            if not has_overlap:
                occupied_ranges_at_level[chosen_level].append(bar_x_range)
                break
            chosen_level += 1
            if chosen_level > 20:
                warnings.warn(f"Превышено максимальное количество уровней (20) для линий значимости.")
                occupied_ranges_at_level.setdefault(chosen_level - 1, []).append(bar_x_range)
                break
        y_mm_for_bar = mm_first_bar + chosen_level * mm_step
        draw_significance_bar_abs(ax, (x1, x2), p_val, y_mm_for_bar, bar_height_mm=bar_height_mm, alpha=alpha)


# ======================================================================================================================
# PLOTTING FUNCTIONS
# ======================================================================================================================
def boxplot_builder(
        file_path: str,
        values_col_idx: int = 1,
        group_col_idx: int = 2,
        subgroup_col_idx: int = None,
        figure_length_key: Literal['1', '3/2', '2', '3'] = '1',
        figure_height_key: Literal['1', '3/2', '2', '3'] = '1'
):

    # Set style
    apply_font_style(plt.gca())

    # Read chart data (sheet 0)
    print(f'Read file: {file_path}')
    try:
        # Read chart data (sheet 0)
        df = pd.read_excel(file_path, sheet_name='Sheet1')

        value_col_name = df.columns[values_col_idx]
        group_col_name = df.columns[group_col_idx]
        if subgroup_col_idx:
            subgroup_col_name = df.columns[subgroup_col_idx]
            subgroup_idx = df[subgroup_col_name]
            subgroups = subgroup_idx.unique()
            stripplot_dodge = True
        else:
            subgroup_idx = None
            subgroups = None
            subgroup_col_name = None
            stripplot_dodge = False

        df[group_col_name] = df[group_col_name].astype(str)
        groups = sorted(df[group_col_name].unique())
        group_sizes = df.groupby(group_col_name).size()

        groups_valid = [g for g in groups if group_sizes[g] >= 5]
        if len(groups_valid) < 2:
            print("Error: a minimum of two cohorts with at least 5 observations is required.")
            return
        print('Data:\n', df.head(), '\n')
    except Exception as e:
        print(f"Data loading error - Sheet0: {e}")
        return

    # Read p-values (sheet 1)
    try:
        pvals = pd.read_excel(file_path, sheet_name=1)
        pvals = pvals.dropna()
        for col in pvals.columns:
            if col != 'p':
                pvals[col] = pvals[col].astype(str)
        print('p-values:', pvals, '\n')
    except Exception as e:
        print(f"Data loading error - Sheet1: {e}")
        return  # Выход из функции

    # Plot the figure
    fig, ax = make_fig_ax(
        length_key=figure_length_key,
        height_key=figure_height_key
    )

    # boxplot
    df_box = df.copy()
    for g, size in group_sizes.items():
        if size < 5:
            df_box.loc[df_box[group_col_name] == g, value_col_name] = np.nan
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
        flierprops=dict(marker='+', markerfacecolor='black', markeredgecolor='black', markersize=3),
        showfliers=True,
        legend=False,
        color='white',
        zorder=1
    )

    # stripplot
    stripplot = sns.swarmplot(
        data=df,
        x=df.columns[group_col_idx],
        y=df.columns[values_col_idx],
        hue=subgroup_idx,
        order=groups,
        hue_order=subgroups,
        dodge=stripplot_dodge,
        palette="Set1",
        legend=True,
        size=3,
        alpha=1,
        ax=ax,
        zorder=0)

    # axes settings
    n_groups = len(groups)
    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlim(-0.5, n_groups - 0.5)
    apply_axes_style(
        ax=ax,
        xlabel=group_col_name,
        ylabel=value_col_name
    )

    # Legend based on stripplot
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    handles, labels = stripplot.get_legend_handles_labels()
    if '' in labels:
        handles = handles[1:]
        labels = labels[1:]
    if handles and labels and len(handles) == len(labels):
        fig.legend(
            handles, labels,
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

    # Draw significance bars for all significant pairs
    draw_significance_bars(
        ax,
        pvals,
        group_col_name,
        subgroup_col_name
    )

    save_file_path = os.path.splitext(file_path)[0] + '.png'
    fig.savefig(save_file_path, dpi=300, transparent=False)
    plt.close(fig)
    print('-'*50)

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

    # version 3.4
    # TODO: colors
    # TODO: s bars
    # TODO: pont shape stripplot
    # TODO: округлить и редуцировать оси (autoscale)
    # TODO: подписи осей X могут не влезать

