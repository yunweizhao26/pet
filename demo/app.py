from bokeh.events import Tap
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div, TextInput, RadioButtonGroup, TextAreaInput, Span, Button, Panel, Tabs
from bokeh.models.tools import CrosshairTool

from demo_utils import (
    get_data,
    prompt_boolq,
    pvp_colors,
    ctl_colors,
    clf_colors,
    reduct,
    task_best_pattern,
    plot_polygons_bokeh,
    advantage_text,
    data_difference,
    calculate_overlap,
    circ_easing,
    average_advantage_text,
    plot_three_polygons_bokeh,
    tasks,
    metric_tap,
    neutral_tasks,
)
from text import text1, text2, text3, text4, initial_passage, initial_question, text5

########################################################################################################################
# Basic dimensions
########################################################################################################################

plot_width = 1200
plot_height = 400
sidebar_width = 400
in_text_plot_height = 300
text_width = 800
widget_size = 400

########################################################################################################################
# Patternification widget
########################################################################################################################

passage = TextAreaInput(title="Passage", rows=3, value=initial_passage, max_width=text_width)
passage.align = "center"
question = TextInput(title="Question", value=initial_question, max_width=text_width)
question.align = "center"
radio_button_group = RadioButtonGroup(labels=["Pattern 1", "Pattern 2", "Pattern 3"], active=0, max_width=text_width)
radio_button_group.align = "center"

box_style = {
    "display": "block",
    "margin": "0 auto",
    "width": f"{text_width}px",
    "text-align": "center",
    "white-space": "pre-wrap",
    "background": "#f4f4f4",
    "border": "1px solid #ddd",
    # "border-left": "3px solid #4d4945",
    "color": "#666",
    "page-break-inside": "avoid",
    # "font-family": "monospace",
    "font-size": "15px",
    "line-height": "1.6",
    "max-width": "100%",
    "overflow": "hidden",
    "min-height": "30px",
    "word-wrap": "break-word",
}

prompt_box = Div(
    text=prompt_boolq(passage.value, question.value, radio_button_group.active),
    width=text_width,
    style=box_style,
    sizing_mode="scale_width",
)
prompt_box.align = "center"


def update_prompt(attrname, old, new):
    prompt_box.text = prompt_boolq(passage.value, question.value, radio_button_group.active)


passage.on_change("value", update_prompt)
question.on_change("value", update_prompt)
radio_button_group.on_change("active", update_prompt)

patternification = column(passage, question, radio_button_group, prompt_box, sizing_mode="scale_width")
patternification.align = "center"

########################################################################################################################
# Advantage diagram
########################################################################################################################

advantage_plots_per_task = []
overlapping_range_per_task = []
training_points_per_task = []
clf_results_per_task = []
pvp_results_per_task = []
advantage_tabs = []
advantage_all_figures = Tabs(tabs=advantage_tabs)

advantage_box = Div(
    text="Click within the comparison region to compute the data advantage for a performance level",
    width=text_width,
    style=box_style,
    sizing_mode="scale_width",
)
advantage_box.align = "center"

for task in tasks:
    training_points, classifier_performances, pattern_performances = get_data(task)
    training_points_per_task.append(list(training_points))
    clf_results_per_task.append(reduct(classifier_performances, "accmax"))
    pvp_results_per_task.append(reduct(pattern_performances, "accmax", task_best_pattern[task], "normal"))
    advantage_plots_per_task.append(plot_polygons_bokeh(
        task, training_points_per_task[-1], clf_results_per_task[-1], pvp_results_per_task[-1], clf_colors,
        pvp_colors
    ))
    advantage_plots_per_task[-1].align = "center"
    advantage_plots_per_task[-1].add_tools(CrosshairTool(dimensions="width", line_alpha=0.2))
    overlapping_range_per_task.append(calculate_overlap(clf_results_per_task[-1], pvp_results_per_task[-1]))
    advantage_tabs.append(Panel(child=advantage_plots_per_task[-1], title=task))

    advantage_plots_per_task[-1].on_event(
        Tap,
        lambda event: metric_tap(
            event,
            overlapping_range_per_task[advantage_all_figures.active],
            training_points_per_task[advantage_all_figures.active],
            clf_results_per_task[advantage_all_figures.active],
            pvp_results_per_task[advantage_all_figures.active],
            advantage_box,
            advantage_plots_per_task[advantage_all_figures.active],
        ),
    )

    if task == "MNLI":
        training_points_per_task.append(list(training_points))
        clf_results_per_task.append(reduct(classifier_performances, "accmax"))
        pvp_results_per_task.append(reduct(pattern_performances, "accmax", task_best_pattern[task], "normal"))
        advantage_plots_per_task.append(plot_polygons_bokeh(
            task, training_points_per_task[-1], clf_results_per_task[-1], pvp_results_per_task[-1], clf_colors,
            pvp_colors, x_log_scale=True
        ))
        advantage_plots_per_task[-1].align = "center"
        advantage_plots_per_task[-1].add_tools(CrosshairTool(dimensions="width", line_alpha=0.2))
        overlapping_range_per_task.append(calculate_overlap(clf_results_per_task[-1], pvp_results_per_task[-1]))
        advantage_tabs.append(Panel(child=advantage_plots_per_task[-1], title="MNLI (log scale)"))

        advantage_plots_per_task[-1].on_event(
            Tap,
            lambda event: metric_tap(
                event,
                overlapping_range_per_task[advantage_all_figures.active],
                training_points_per_task[advantage_all_figures.active],
                clf_results_per_task[advantage_all_figures.active],
                pvp_results_per_task[advantage_all_figures.active],
                advantage_box,
                advantage_plots_per_task[advantage_all_figures.active],
            ),
        )

advantage_all_figures = Tabs(tabs=advantage_tabs)
advantage_all_figures.align = "center"


def on_integrate_click():
    frames = 200
    initial_placement = overlapping_range_per_task[advantage_all_figures.active][0]

    if not isinstance(advantage_plots_per_task[advantage_all_figures.active].renderers[-1], Span):
        metric_line = Span(
            location=initial_placement,
            line_alpha=0.7,
            dimension="width",
            line_color=clf_colors[0] if initial_placement < 0 else pvp_colors[0],
            line_dash="dashed",
            line_width=1,
        )
        advantage_plots_per_task[advantage_all_figures.active].renderers.extend([metric_line])
    else:
        advantage_plots_per_task[advantage_all_figures.active].renderers[-1].location = initial_placement
        advantage_plots_per_task[advantage_all_figures.active].renderers[-1].line_color = clf_colors[
            0] if initial_placement < 0 else pvp_colors[0]

    average_advantage = 0
    for i in range(1, frames):
        metric_value = overlapping_range_per_task[advantage_all_figures.active][0] + (
                overlapping_range_per_task[advantage_all_figures.active][1] -
                overlapping_range_per_task[advantage_all_figures.active][0]) * (i / frames)
        advantage_value = data_difference(metric_value, overlapping_range_per_task[advantage_all_figures.active],
                                          training_points_per_task[advantage_all_figures.active],
                                          clf_results_per_task[advantage_all_figures.active],
                                          pvp_results_per_task[advantage_all_figures.active])
        average_advantage = ((i - 1) * average_advantage + advantage_value) / i

        advantage_plots_per_task[advantage_all_figures.active].renderers[-1].location = metric_value
        advantage_plots_per_task[advantage_all_figures.active].renderers[-1].line_color = clf_colors[
            0] if advantage_value < 0 else pvp_colors[0]
        advantage_box.text = average_advantage_text(average_advantage)


integrate = Button(width=175, max_width=175, label="Integrate over the whole region!")
integrate.align = "center"
integrate.on_click(on_integrate_click)


def on_tab_change(attr, old, new):
    advantage_box.text = "Click within the comparison region to compute the data advantage for a performance level"


advantage_all_figures.on_change('active', on_tab_change)

advantage_column = column(advantage_all_figures, advantage_box, integrate, sizing_mode="scale_width")

########################################################################################################################
# Null verbalizer diagram
########################################################################################################################

null_tabs = []
null_all_figures = Tabs(tabs=null_tabs)

for task in neutral_tasks:
    training_points, classifier_performances, pattern_performances = get_data(task)
    training_points = list(training_points)
    clf_results = reduct(classifier_performances, "accmax")
    pvp_results = reduct(pattern_performances, "accmax", task_best_pattern[task], "normal")
    ctl_results = reduct(pattern_performances, "accmax", task_best_pattern[task], "neutral")
    null_plot = plot_three_polygons_bokeh(task, training_points, clf_results, pvp_results, ctl_results, clf_colors,
                                          pvp_colors, ctl_colors)
    null_plot.align = "center"
    null_plot.add_tools(CrosshairTool(dimensions="width", line_alpha=0.2))
    null_tabs.append(Panel(child=null_plot, title=task))

    if task == "MNLI":
        null_plot = plot_three_polygons_bokeh(task, training_points, clf_results, pvp_results, ctl_results, clf_colors,
                                              pvp_colors, ctl_colors, x_log_scale=True)
        null_plot.align = "center"
        null_plot.add_tools(CrosshairTool(dimensions="width", line_alpha=0.2))
        null_tabs.append(Panel(child=null_plot, title="MNLI (log scale)"))

null_all_figures = Tabs(tabs=null_tabs)
null_all_figures.align = "center"

########################################################################################################################
# Add write-up text
########################################################################################################################

main_text_style = {
    "min-height": "100px",
    "overflow": "hidden",
    "display": "block",
    "margin": "auto",
    "width": f"{text_width}px",
    "font-size": "18px",
}

textbox1 = Div(text=text1, style=main_text_style)
textbox2 = Div(text=text2, style=main_text_style)
textbox3 = Div(text=text3, style=main_text_style)
textbox4 = Div(text=text4, style=main_text_style)
textbox5 = Div(text=text5, style=main_text_style)
textbox1.align = "center"
textbox2.align = "center"
textbox3.align = "center"
textbox4.align = "center"
textbox5.align = "center"

########################################################################################################################
# Set up layouts and add to document
########################################################################################################################

main_body = column(textbox1, patternification, textbox2, advantage_column, textbox3, null_all_figures, textbox4,
                   textbox5, sizing_mode="scale_width")
main_body.align = "center"

curdoc().add_root(main_body)
curdoc().title = "How many data points is a prompt worth ?"
