import math

import pandas as pd
import numpy as np
from itertools import product
import shapely
from bokeh.models import Span, Label, ColumnDataSource, Whisker
from bokeh.plotting import figure, show
from shapely.geometry import Polygon
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn

task_patterns = {
    "CB": [0, 3],
    "RTE": [0, 3],
    "BoolQ": [0, 3, 5],
    "MNLI": [0, 3],
    "COPA": [0, 1],
    "WSC": [0, 1, 2],
    "WiC": [0, 1],
    "MultiRC": [0, 1, 2],
}
task_reps = {"CB": 4, "RTE": 4, "BoolQ": 4, "MNLI": 4, "COPA": 4, "WSC": 4, "WiC": 4, "MultiRC": 4}
task_best_pattern = {"CB": 0, "RTE": 0, "BoolQ": 0, "MNLI": 0, "COPA": 1, "WSC": 0, "WiC": 0, "MultiRC": 1}
task_metric_short = {
    "CB": "f1-macro",
    "RTE": "acc",
    "BoolQ": "acc",
    "MNLI": "acc",
    "COPA": "acc",
    "WSC": "acc",
    "WiC": "acc",
    "MultiRC": "f1",
}
task_metrics = {
    "CB": "F1-macro",
    "RTE": "accuracy",
    "BoolQ": "accuracy",
    "MNLI": "accuracy",
    "COPA": "accuracy",
    "WSC": "accuracy",
    "WiC": "accuracy",
    "MultiRC": "F1",
}
task_neutral = {
    "CB": True,
    "RTE": True,
    "BoolQ": True,
    "MNLI": True,
    "COPA": False,
    "WSC": False,
    "multirc": True,
    "WiC": True,
    "MultiRC": True,
}
neutral_tasks = [
    "BoolQ",
    "CB",
    "MNLI",
    "MultiRC",
    "RTE",
    "WiC",
]
tasks = sorted(task_patterns.keys())

pvp_colors = ["goldenrod", "blanchedalmond", "floralwhite"]
ctl_colors = ["crimson", "salmon", "mistyrose"]
clf_colors = ["indigo", "plum", "thistle"]


def prompt_boolq(passage, question, pattern):
    if pattern == 0:
        return f"""<span style="color: #0c593d">{passage}</span> <span style="color: #910713"><b>Based on the previous passage,</b></span> <span style="color: #031154">{question}</span> <span style="color: #ba9004"><b>[YES/NO]</b></span>"""
    if pattern == 1:
        return f"""<span style="color: #0c593d">{passage}</span><span style="color: #910713"><b> Question:</b></span> <span style="color: #031154">{question}</span><span style="color: #910713"><b> Answer: </b></span><span style="color: #ba9004"><b>[YES/NO]</b></span>"""
    if pattern == 2:
        return f"""<span style="color: #910713"><b>Based on the following passage,</b></span> <span style="color: #031154">{question}</span><span style="color: #ba9004"><b> [YES/NO]</b></span> <span style="color: #0c593d">{passage}</span>"""


def advantage_text(advantage):
    model_type = (
        """<span style="color: #4B0082">Head</span>"""
        if advantage < 0
        else """<span style="color: #daa520">Prompting</span>"""
    )
    return f"""<b>{model_type}</b> advantage: <b>{abs(advantage):.2f}</b> data points"""


def average_advantage_text(advantage):
    model_type = (
        """<span style="color: #4B0082">head</span>"""
        if advantage < 0
        else """<span style="color: #daa520">prompting</span>"""
    )
    return f"""<b>Average {model_type}</b> advantage: <b>{abs(advantage):.2f}</b> data points"""


def naming_convention(task, seed, pvp_index=None, neutral=False):
    method = f"PVP {pvp_index}" if pvp_index is not None else "CLF"
    model = "roberta"
    if neutral:
        verbalizer = "neutral"
    else:
        verbalizer = None
    return (
            f"{method} {model}"
            + (f" {verbalizer} verbalizer" if verbalizer is not None else "")
            + f" seed {seed} - test-{task_metric_short[task]}-all-p"
    )


def get_data(task):
    url = f"https://raw.githubusercontent.com/TevenLeScao/pet/master/exported_results/{task.lower()}/wandb_export.csv"
    df = pd.read_csv(url)
    training_points = df["training_points"]

    head_performances = np.transpose(np.array([df[naming_convention(task, i)] for i in range(task_reps[task])]))
    pattern_performances = {}
    for pattern in task_patterns[task]:
        pattern_performances[pattern] = {
            "normal": np.transpose(np.array([df[naming_convention(task, i, pattern)] for i in range(task_reps[task])]))
        }
        if task_neutral[task]:
            pattern_performances[pattern]["neutral"] = np.transpose(
                np.array([df[naming_convention(task, i, pattern, True)] for i in range(task_reps[task])])
            )

    return training_points, head_performances, pattern_performances


def reduct(performances, reduction="accmax", final_pattern=0, verbalizer="normal", exclude=None):
    # Combining the different runs for each experimental set-up
    reducted = None

    if isinstance(performances, dict):
        performances = performances[final_pattern][verbalizer]
    if exclude is not None:
        performances = np.delete(performances, exclude, axis=1)

    if reduction == "avg":
        # Average
        reducted = np.nanmean(performances, axis=1)

    if reduction == "std":
        # Standard deviation
        reducted = np.nanstd(performances, axis=1)

    if reduction == "max":
        # Maximum
        reducted = np.nanmax(performances, axis=1)

    if reduction == "accmax":
        # This makes the maximum curve monotonic
        max_performance = np.nanmax(performances, axis=1)
        reducted = np.maximum.accumulate(max_performance)

    assert reducted is not None, "unrecognized reduction method"
    return reducted


def find_surrounding_points(perf, clf_results, pvp_results):
    for i, clf_result in enumerate(clf_results):
        if i - 1 > 0 and clf_result == clf_results[i - 1]:
            continue
        if clf_result > perf:
            if i == 0:
                raise ValueError(f"value {perf} too small")
            else:
                break
    for j, pvp_result in enumerate(pvp_results):
        if j - 1 > 0 and pvp_result == pvp_results[j - 1]:
            continue
        if pvp_result > perf:
            if j == 0:
                raise ValueError(f"value {perf} too small")
            else:
                break
    return i - 1, j - 1


def interpolate(perf, x1, x2, y1, y2):
    return x1 + (perf - y1) * (x2 - x1) / (y2 - y1)


def interpolate_from_idx(perf, idx, results, training_points):
    return interpolate(perf, training_points[idx], training_points[idx + 1], results[idx], results[idx + 1])


def interpolate_from_perf(perf, overlapping_range, training_points, clf_results, pvp_results):
    if not overlapping_range[0] <= perf <= overlapping_range[1]:
        raise ValueError(f"perf {perf} not in acceptable bounds {overlapping_range}")
    clf_idx, pvp_idx = find_surrounding_points(perf, clf_results, pvp_results)
    return interpolate_from_idx(perf, clf_idx, clf_results, training_points), interpolate_from_idx(
        perf, pvp_idx, pvp_results, training_points
    )


def data_difference(perf, overlapping_range, training_points, clf_results, pvp_results):
    x1, x2 = interpolate_from_perf(perf, overlapping_range, training_points, clf_results, pvp_results)
    return x1 - x2


def calculate_overlap(clf_results, pvp_results, full_range=False):
    if full_range:
        return (min(min(clf_results), min(pvp_results)), max(max(clf_results), max(pvp_results)))
    else:
        return (max(min(clf_results), min(pvp_results)), min(max(clf_results), max(pvp_results)))


def calculate_range(overlapping_range, number_of_points):
    integral_range = (
        overlapping_range[0] + i / (number_of_points + 1) * (overlapping_range[1] - overlapping_range[0])
        for i in range(1, number_of_points + 1)
    )
    return integral_range


def calculate_differences(integral_range, overlapping_range, training_points, clf_results, pvp_results):
    differences = [
        data_difference(y, overlapping_range, training_points, clf_results, pvp_results) for y in integral_range
    ]
    return differences


def calculate_offset(training_points, clf_results, pvp_results, number_of_points=1000):
    overlapping_range = calculate_overlap(clf_results, pvp_results)
    integral_range = calculate_range(overlapping_range, number_of_points)
    differences = calculate_differences(integral_range, overlapping_range, training_points, clf_results, pvp_results)
    offset = sum(differences) / number_of_points
    return offset


def intersection_with_range(training_points, results, band):
    result_polygon = Polygon(
        [(training_points[i], results[i]) for i in range(len(training_points))]
        + [(training_points[-1], 0), (training_points[0], 0)]
    )
    return result_polygon.intersection(band)


def fill_polygon(fig, polygon, color, label=None, alpha=1.0):
    if polygon.is_empty or isinstance(polygon, shapely.geometry.LineString):
        return
    if isinstance(polygon, Polygon):
        xs, ys = polygon.exterior.xy
        fig.patch(xs, ys, color=color, alpha=alpha)
    else:
        for geom in polygon.geoms:
            if isinstance(geom, shapely.geometry.LineString):
                continue
            xs, ys = geom.exterior.xy
            fig.patch(xs, ys, color=color, alpha=alpha)
            label = None


label_order = {
    "head run": 0,
    "head advantage": 1,
    "control run": 2,
    "optimization advantage": 3,
    "prompting run": 4,
    "semantics advantage": 5,
    "region of comparison": 6,
}


def metric_tap(
        event, overlapping_range, training_points, clf_results, pvp_results, advantage_box, advantage_plot
):
    _, metric_value = event.x, event.y
    try:
        advantage_value = data_difference(metric_value, overlapping_range, training_points, clf_results, pvp_results)
        advantage_box.text = advantage_text(advantage_value)
        if not isinstance(advantage_plot.renderers[-1], Span):
            metric_line = Span(
                location=metric_value,
                line_alpha=0.7,
                dimension="width",
                line_color=clf_colors[0] if advantage_value < 0 else pvp_colors[0],
                line_dash="dashed",
                line_width=1,
            )
            advantage_plot.renderers.extend([metric_line])
        else:
            advantage_plot.renderers[-1].location = metric_value
            advantage_plot.renderers[-1].line_color = clf_colors[0] if advantage_value < 0 else pvp_colors[0]
    # clicking outside the region
    except ValueError:
        pass


def plot_polygons_bokeh(task, training_points, clf_results, pvp_results, clf_colors, pvp_colors, x_log_scale=False):
    overlapping_range = calculate_overlap(clf_results, pvp_results, False)
    full_range = calculate_overlap(clf_results, pvp_results, True)
    middle_y = (full_range[0] + full_range[1]) / 2

    fig = figure(plot_height=400, plot_width=800, max_height=400, max_width=800,
                 x_axis_type="log" if x_log_scale else "linear", title="Performance over training subset sizes of head and prompting methods")

    fig.circle(training_points, clf_results, color=clf_colors[0], legend="head run")
    fig.circle(training_points, pvp_results, color=pvp_colors[0], legend="prompting run")
    fig.line(training_points, clf_results, color=clf_colors[0], alpha=1)
    fig.line(training_points, pvp_results, color=pvp_colors[0], alpha=1)
    fig.xaxis.axis_label = "training subset size"
    fig.yaxis.axis_label = task_metrics[task]
    fig.patch(
        [training_points[0], training_points[0], training_points[-1], training_points[-1]],
        [overlapping_range[0], overlapping_range[1], overlapping_range[1], overlapping_range[0]],
        color="black",
        fill_alpha=0,
        line_width=0,
        legend="comparison region",
        hatch_alpha=0.14,
        hatch_scale=40,
        hatch_pattern="/",
    )

    band = Polygon(
        [
            (training_points[0], overlapping_range[0]),
            (training_points[0], overlapping_range[1]),
            (training_points[-1], overlapping_range[1]),
            (training_points[-1], overlapping_range[0]),
        ]
    )
    full_band = Polygon(
        [
            (training_points[0], full_range[0]),
            (training_points[0], full_range[1]),
            (training_points[-1], full_range[1]),
            (training_points[-1], full_range[0]),
        ]
    )
    clf_polygon = intersection_with_range(training_points, clf_results, band)
    pvp_polygon = intersection_with_range(training_points, pvp_results, band)
    full_clf_polygon = intersection_with_range(training_points, clf_results, full_band)
    full_pvp_polygon = intersection_with_range(training_points, pvp_results, full_band)

    clf_inside_area = clf_polygon.difference(pvp_polygon)
    pvp_inside_area = pvp_polygon.difference(clf_polygon)
    clf_outside_area = (full_clf_polygon.difference(full_pvp_polygon)).difference(clf_inside_area)
    pvp_outside_area = (full_pvp_polygon.difference(full_clf_polygon)).difference(pvp_inside_area)

    fill_polygon(fig, clf_outside_area, clf_colors[1], alpha=0.13)
    fill_polygon(fig, pvp_outside_area, pvp_colors[1], alpha=0.18)
    fill_polygon(
        fig, clf_inside_area, clf_colors[1], alpha=0.4, label="head advantage" if task == "WiC" else None
    )
    fill_polygon(fig, pvp_inside_area, pvp_colors[1], alpha=0.4, label="prompting advantage")

    fig.line([training_points[0], training_points[-1]], [overlapping_range[0], overlapping_range[0]], color="dimgrey")
    fig.line([training_points[0], training_points[-1]], [overlapping_range[1], overlapping_range[1]], color="dimgrey")

    vline = Span(
        location=training_points[-1], dimension="height", line_color="black", line_width=2.5, line_dash="dashed"
    )
    end_label = Label(
        x=training_points[-1], y=middle_y, text="End of dataset", angle=90, angle_units="deg", text_align="center"
    )
    fig.renderers.extend([vline, end_label])

    fig.legend.location = "bottom_right"

    return fig


def plot_three_polygons_bokeh(
        task, training_points, clf_results, pvp_results, ctl_results, clf_colors, pvp_colors, ctl_colors,
        x_log_scale=False
):
    overlapping_range = calculate_overlap(clf_results, pvp_results, False)
    full_range = calculate_overlap(clf_results, pvp_results, True)
    middle_y = (full_range[0] + full_range[1]) / 2

    fig = figure(plot_height=400, plot_width=800, max_height=400, max_width=800,
                 x_axis_type="log" if x_log_scale else "linear", title="Performance over training subset sizes of head, prompting and prompting with a null verbalizer")
    fig.xaxis.axis_label = "training subset size"
    fig.yaxis.axis_label = task_metrics[task]
    fig.circle(training_points, clf_results, color=clf_colors[0], legend="head run")
    fig.circle(training_points, pvp_results, color=pvp_colors[0], legend="prompting run")
    fig.circle(training_points, ctl_results, color=ctl_colors[0], legend="null verbalizer run")
    fig.line(training_points, clf_results, color=clf_colors[0], alpha=1)
    fig.line(training_points, pvp_results, color=pvp_colors[0], alpha=1)
    fig.line(training_points, ctl_results, color=ctl_colors[0], alpha=1)

    fig.patch(
        [training_points[0], training_points[0], training_points[-1], training_points[-1]],
        [overlapping_range[0], overlapping_range[1], overlapping_range[1], overlapping_range[0]],
        color="black",
        fill_alpha=0,
        line_width=0,
        legend="comparison region",
        hatch_alpha=0.14,
        hatch_scale=40,
        hatch_pattern="/",
    )

    band = Polygon(
        [
            (training_points[0], overlapping_range[0]),
            (training_points[0], overlapping_range[1]),
            (training_points[-1], overlapping_range[1]),
            (training_points[-1], overlapping_range[0]),
        ]
    )
    full_band = Polygon(
        [
            (training_points[0], full_range[0]),
            (training_points[0], full_range[1]),
            (training_points[-1], full_range[1]),
            (training_points[-1], full_range[0]),
        ]
    )

    clf_polygon = intersection_with_range(training_points, clf_results, band)
    pvp_polygon = intersection_with_range(training_points, pvp_results, band)
    ctl_polygon = intersection_with_range(training_points, ctl_results, band)

    full_clf_polygon = intersection_with_range(training_points, clf_results, full_band)
    full_pvp_polygon = intersection_with_range(training_points, pvp_results, full_band)
    full_ctl_polygon = intersection_with_range(training_points, ctl_results, full_band)

    clf_inside_area = clf_polygon.difference(ctl_polygon)
    pvp_inside_area = pvp_polygon.difference(clf_polygon).difference(ctl_polygon)
    ctl_inside_area = ctl_polygon.difference(clf_polygon)

    clf_outside_area = (full_clf_polygon.difference(full_ctl_polygon)).difference(clf_inside_area)
    pvp_outside_area = (full_pvp_polygon.difference(full_clf_polygon).difference(ctl_polygon)).difference(
        pvp_inside_area
    )
    ctl_outside_area = (full_ctl_polygon.difference(full_clf_polygon)).difference(pvp_inside_area)

    fill_polygon(
        fig, clf_inside_area, clf_colors[1], alpha=0.4, label="head advantage" if task == "WiC" else None
    )
    fill_polygon(fig, pvp_inside_area, pvp_colors[1], alpha=0.4, label="prompting advantage")
    fill_polygon(fig, ctl_inside_area, ctl_colors[1], alpha=0.4, label="null verbalizer advantage")
    fill_polygon(fig, clf_outside_area, clf_colors[1], alpha=0.13)
    fill_polygon(fig, pvp_outside_area, pvp_colors[1], alpha=0.18)
    fill_polygon(fig, ctl_outside_area, ctl_colors[1], alpha=0.13)

    fig.line([training_points[0], training_points[-1]], [overlapping_range[0], overlapping_range[0]], color="dimgrey")
    fig.line([training_points[0], training_points[-1]], [overlapping_range[1], overlapping_range[1]], color="dimgrey")

    vline = Span(
        location=training_points[-1], dimension="height", line_color="black", line_width=2.5, line_dash="dashed"
    )
    end_label = Label(
        x=training_points[-1], y=middle_y, text="End of dataset", angle=90, angle_units="deg", text_align="center"
    )
    fig.renderers.extend([vline, end_label])

    fig.legend.location = "bottom_right"

    return fig


def pattern_graph(task):
    fig = figure(plot_height=400, plot_width=800, max_height=400, max_width=800, x_axis_type="log", title="Performance over training subset sizes of different prompt patterns")
    fig.xaxis.axis_label = "training subset size"
    fig.yaxis.axis_label = task_metrics[task]
    url = f"https://raw.githubusercontent.com/TevenLeScao/pet/master/exported_results/{task.lower()}/wandb_export.csv"
    df = pd.read_csv(url)
    expanded_training_points = np.array(list(df["training_points"]) * task_reps[task] * len(task_patterns[task]))
    data = np.array(df[[naming_convention(task, seed, pattern) for pattern in task_patterns[task] for seed in
                        range(task_reps[task])]])
    data = data.reshape(-1, task_reps[task])
    col_med = np.nanmean(data, axis=1)
    # Find indices that you need to replace
    inds = np.where(np.isnan(data))
    # Place column means in the indices. Align the arrays using take
    data[inds] = np.take(col_med, inds[0])
    data = data.reshape(len(df["training_points"]), -1)
    data = data.transpose().reshape(-1)
    data = data + np.random.normal(0, 0.01, len(data))
    pattern = np.array([i // (len(data) // len(task_patterns[task])) for i in range(len(data))])
    seed = np.array([0, 1, 2, 3] * (len(data) // task_reps[task]))
    long_df = pd.DataFrame(np.stack((expanded_training_points, pattern, seed, data), axis=1),
                           columns=["training_points", "pattern", "seed", task_metrics[task]])
    long_df['pattern'] = long_df['pattern'].astype(int).astype(str)
    gby_pattern = long_df.groupby('pattern')
    pattern_colors = ["royalblue", "darkturquoise", "darkviolet"]

    for i, (pattern, pattern_df) in enumerate(gby_pattern):
        gby_training_points = pattern_df.groupby('training_points')
        x = [training_point for training_point, training_point_df in gby_training_points]
        y_max = list([np.max(training_point_df[task_metrics[task]]) for training_point, training_point_df in gby_training_points])
        y_min = list([np.min(training_point_df[task_metrics[task]]) for training_point, training_point_df in gby_training_points])
        y = list([np.median(training_point_df[task_metrics[task]]) for training_point, training_point_df in gby_training_points])
        fig.circle(x, y, color=pattern_colors[i], alpha=1, legend=f"Pattern {i}")
        fig.line(x, y, color=pattern_colors[i], alpha=1)
        fig.varea(x=x, y1=y_max, y2=y_min, color=pattern_colors[i], alpha=0.11)
        # source = ColumnDataSource(data=dict(base=x, lower=y_min, upper=y_max))
        # w = Whisker(source=source, base="base", upper="upper", lower="lower", line_color=pattern_colors[i], line_alpha=0.3)
        # w.upper_head.line_color = pattern_colors[i]
        # w.lower_head.line_color = pattern_colors[i]
        # fig.add_layout(w)

    return fig



def cubic_easing(t):
    if t < 0.5:
        return 4 * t * t * t
    p = 2 * t - 2
    return 0.5 * p * p * p + 1


def circ_easing(t):
    if t < 0.5:
        return 0.5 * (1 - math.sqrt(1 - 4 * (t * t)))
    return 0.5 * (math.sqrt(-((2 * t) - 3) * ((2 * t) - 1)) + 1)
