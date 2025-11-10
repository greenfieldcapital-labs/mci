import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from tqdm import tqdm

from mci.core.minima import find_local_minima_with_separation
from mci.forecasting.predictions import generate_sudo_predictions_for_frame

lm_palette = plotly.colors.qualitative.Safe  # or Set1, D3, etc.


def get_color(val):
    # Custom color interpolation: 0 -> #FF8F6C (red), 50 -> #FFE900 (yellow), 100 -> #02FF47 (green)
    if val <= 50:
        t = val / 50.0
        r = int(255 * (1 - t) + 255 * t)
        g = int(143 * (1 - t) + 233 * t)
        b = int(108 * (1 - t) + 0 * t)
    else:
        t = (val - 50) / 50.0
        r = int(255 * (1 - t) + 2 * t)
        g = int(233 * (1 - t) + 255 * t)
        b = int(0 * (1 - t) + 71 * t)

    # Clamp RGB values to valid range [0, 255]
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f"rgb({r},{g},{b})"


trace_predctions = 'predictions'
trace_avg_pred = 'avg_prediction'
trace_local_min = 'local_min'


# Animation Prep
def plot_result(
    df,
    df_line,
    df_line_col,
    horizon = 180,

    N = 14,
    min_separation = 7,
    top_k = 5,
    prediction_averaging_range = 3,

    show_scatter = True,
    calc_every_nth = 1,
    title = "Animated distances and sudo prediction lines",
    frame_duration=1e3,
    transition_duration=3e2,
    add_traces = None
):
    if add_traces is None:
        add_traces = [
            trace_predctions,
            trace_avg_pred,
            trace_local_min
            
        ]
    df = df.copy()
    df.dist -= df.dist.min()
    dist_max = df.dist.max()
    if dist_max > 0:
        df.dist /= dist_max
    else:
        # If all distances are the same (constant), set to middle value
        df.dist = 0.5
    df.dist *= 100

    df = df.sort_values(['current_date', 'max_date']).reset_index(drop=True)
    df_line = df_line.sort_values('date').reset_index(drop=True)
    current_dates = sorted(df['current_date'].unique())
    max_x_pred = max(current_dates) + pd.Timedelta(days=horizon)
    x_range = [df['max_date'].min(), max(df['max_date'].max(), df_line['date'].max(), max_x_pred)]
    global_min_dist = df['dist'].min()
    global_max_dist = df['dist'].max()
    y1_range = [0, 100]
    y2_range = [df_line[df_line_col].min(), df_line[df_line_col].max()]

    lms_df = find_local_minima_with_separation(
        df, N=N, min_separation=min_separation, top_k=top_k,
    )
    if lms_df.empty:
        lms_by_frame = {cd: pd.DataFrame(columns=['current_date', 'max_date', 'dist']) for cd in current_dates}
    else:
        lms_by_frame = {cd: lms_df[lms_df['current_date'] == cd] for cd in current_dates}

    # Calculate max_lms across framed current_dates
    max_lms = 0
    for curr_date in current_dates[0::3]:
        lms_this = lms_by_frame.get(curr_date, pd.DataFrame())
        max_lms = max(max_lms, len(lms_this))

    # Calculate y2 range to fit all sudo predictions
    all_pred_predicted_values = []
    for curr_date in tqdm(current_dates[0::calc_every_nth]):
        lms_this = lms_by_frame.get(curr_date, pd.DataFrame())
        if not lms_this.empty:
            lms_this = lms_this.sort_values('dist').reset_index(drop=True)
        preds = generate_sudo_predictions_for_frame(
            curr_date,
            lms_this,
            df_line,
            horizon=horizon,
            prediction_averaging_range=prediction_averaging_range,
            df_line_col=df_line_col
        )
        for pred in preds:
            all_pred_predicted_values.extend([p for p in pred['predicted_values'] if not np.isnan(p)])
    if all_pred_predicted_values:
        y2_range = [min(y2_range[0], min(all_pred_predicted_values)), max(y2_range[1], max(all_pred_predicted_values))]


    frames = []
    for curr_date in tqdm(current_dates[0::calc_every_nth]):
        frame_data = []
        dff = df[df["current_date"] == curr_date]
        similarity = 100 * (global_max_dist - dff['dist']) / (global_max_dist - global_min_dist + 1e-8)
        sctr = None
        if show_scatter:
            sctr = go.Scatter(
                x=dff["max_date"],
                y=similarity,
                mode="markers",
                marker=dict(
                    color=[get_color(v) for v in similarity],
                    size=8,
                ),
                name="Match %",
                yaxis="y1",
                showlegend=(curr_date == current_dates[0]),
                hovertemplate="Date: %{x|%Y-%m-%d}<br>Match: %{y:.1f}%"
            )
            frame_data.append(sctr)
        predicted_values_line = go.Scatter(
            x=df_line["date"],
            y=df_line[df_line_col],
            mode="lines",
            line=dict(color="#ECECE4", width=4),  # Increased thickness for better readability
            name=f"actual data for {df_line_col}",
            yaxis="y2",
            showlegend=(curr_date == current_dates[0])
        )
        frame_data.append(predicted_values_line)

        # Sudo prediction lines and LM markers with matching color
        lms_this_frame = lms_by_frame.get(curr_date, pd.DataFrame())
        if not lms_this_frame.empty:
            lms_this_frame = lms_this_frame.sort_values('dist').reset_index(drop=True)
        predictions = generate_sudo_predictions_for_frame(
            curr_date,
            lms_this_frame,
            df_line,
            horizon=horizon,
            prediction_averaging_range=prediction_averaging_range,
            df_line_col=df_line_col
        )

        num_lms = len(lms_this_frame)
        lm_color_dict = {}
        if num_lms > 0:
            start_color = (255, 233, 0)  # #FFE900
            end_color = (191, 155, 222)  # #BF9BDE
            colors = []
            for i in range(num_lms):
                t = i / (num_lms - 1) if num_lms > 1 else 0
                r = int(start_color[0] * (1 - t) + end_color[0] * t)
                g = int(start_color[1] * (1 - t) + end_color[1] * t)
                b = int(start_color[2] * (1 - t) + end_color[2] * t)
                colors.append(f"rgb({r},{g},{b})")
            lm_color_dict = {lms_this_frame.loc[i, 'max_date']: colors[i] for i in range(num_lms)}

        all_pred_arrays = []
        for idx in range(max_lms):
            if idx < num_lms:
                pred = predictions[idx]
                lm_date = lms_this_frame.loc[idx]['max_date']
                color = lm_color_dict[lm_date]
                x_pred = [curr_date + pd.Timedelta(days=k+1) for k in range(horizon)]
                legend_name = f"Pred: {lm_date.date()}"
                line = go.Scatter(
                    x=x_pred,
                    y=pred['predicted_values'],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=legend_name,
                    yaxis="y2",
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"Prediction from ({idx}): {lm_date.date()}<br>%{{x}}: %{{y}}"
                )
                if trace_predctions in add_traces:
                    frame_data.append(line)
                all_pred_arrays.append(pred['predicted_values'])
                # LM vertical line ON predicted_values (y2)
                lm_row = lms_this_frame.loc[idx]
                predicted_values_at_lm = df_line[df_line["date"] == lm_row["max_date"]][df_line_col]
                predicted_values_val = predicted_values_at_lm.iloc[0] if not predicted_values_at_lm.empty else y2_range[0]
                lm_vline = go.Scatter(
                    x=[lm_row["max_date"], lm_row["max_date"]],
                    y=y2_range,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"LM {lm_row['max_date'].date()}",
                    yaxis="y2",
                    showlegend=False
                )
                if trace_local_min in add_traces:
                    frame_data.append(lm_vline)
                # Add X mark at the top of the line
                lm_xmark = go.Scatter(
                    x=[lm_row["max_date"]],
                    y=[predicted_values_val],
                    mode="markers+text",  # Combine markers (for rounded box) and text (for number)
                    marker=dict(
                        color="white",  # Filled white for the box
                        size=24,  # Adjust size to control box diameter (larger for bigger box)
                        symbol="circle",  # Rounded shape
                        line=dict(color="black", width=1)  # Optional border for definition
                    ),
                    text=[str(idx+1)],  # Replace 'number' with actual integer (e.g., 1, 2, ... N); define dynamically
                    textposition="middle center",  # Center text inside the circle
                    textfont=dict(color="black", size=14),  # Black for visibility on white; adjust size as needed
                    name=f"LM {lm_row['max_date'].date()}",
                    yaxis="y2",
                    showlegend=False
                )
                if trace_local_min in add_traces:
                    frame_data.append(lm_xmark)
            else:
                pred = predictions[-1]
                lm_date = lms_this_frame.loc[num_lms-1]['max_date']
                color = lm_color_dict[lm_date]
                x_pred = [curr_date + pd.Timedelta(days=k+1) for k in range(horizon)]
                legend_name = f"Pred: {lm_date.date()}"
                line = go.Scatter(
                    x=x_pred,
                    y=pred['predicted_values'],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=legend_name,
                    yaxis="y2",
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate=f"Prediction from ({idx}): {lm_date.date()}<br>%{{x}}: %{{y}}"
                )
                if trace_predctions in add_traces:
                    frame_data.append(line)
                # all_pred_arrays.append(pred['predicted_values'])
                # LM vertical line ON predicted_values (y2)
                lm_row = lms_this_frame.loc[num_lms-1]
                predicted_values_at_lm = df_line[df_line["date"] == lm_row["max_date"]][df_line_col]
                predicted_values_val = predicted_values_at_lm.iloc[0] if not predicted_values_at_lm.empty else y2_range[0]
                lm_vline = go.Scatter(
                    x=[lm_row["max_date"], lm_row["max_date"]],
                    y=y2_range,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"LM {lm_row['max_date'].date()}",
                    yaxis="y2",
                    showlegend=False
                )
                if trace_local_min in add_traces:
                    frame_data.append(lm_vline)
                lm_xmark = go.Scatter(
                    x=[lm_row["max_date"]],
                    y=[predicted_values_val],
                    mode="markers+text",  # Combine markers (for rounded box) and text (for number)
                    marker=dict(
                        color="white",  # Filled white for the box
                        size=24,  # Adjust size to control box diameter (larger for bigger box)
                        symbol="circle",  # Rounded shape
                        line=dict(color="black", width=1)  # Optional border for definition
                    ),
                    text=[str(idx+1)],  # Replace 'number' with actual integer (e.g., 1, 2, ... N); define dynamically
                    textposition="middle center",  # Center text inside the circle
                    textfont=dict(color="black", size=14),  # Black for visibility on white; adjust size as needed
                    name=f"LM {lm_row['max_date'].date()}",
                    yaxis="y2",
                    showlegend=False
                )
                if trace_local_min in add_traces:
                    frame_data.append(lm_xmark)


        # --- NEW: Add average forecast line ---

        epsilon = 1e-8


        if all_pred_arrays:
            arr = np.array(all_pred_arrays)  # shape: [n_preds, horizon]

            conf_color = "#ECECE4"
            # Use conf_color for the average forecast line:
            dists = lms_this_frame['dist'].values
            weights = 1 / (dists + epsilon)
            weights /= weights.sum()  # Normalize

            # Weighted average (instead of np.nanmean)
            avg_pred = np.average(arr, axis=0, weights=weights)


            x_pred = [curr_date + pd.Timedelta(days=k+1) for k in range(horizon)]
            avg_line = go.Scatter(
                x=x_pred,
                y=avg_pred,
                mode="lines",
                line=dict(color=conf_color, width=4, dash="dash"),
                name="Average forecast",
                yaxis="y2",
                opacity=1.0,
                showlegend=False
            )
            if trace_avg_pred in add_traces:
                frame_data.append(avg_line)
        else:
            placeholder_avg = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(width=4, dash="dash"),
                yaxis="y2",
                visible=False,
                showlegend=False
            )
            if trace_avg_pred in add_traces:
                frame_data.append(placeholder_avg)


        predicted_values_at_curr = df_line[df_line["date"] == curr_date][df_line_col]
        curr_vline = go.Scatter()
        x_marker = go.Scatter()
        if not predicted_values_at_curr.empty:
            predicted_values_val = predicted_values_at_curr.iloc[0]
            # Vertical line for current date spanning entire y2
            curr_vline = go.Scatter(
                x=[curr_date, curr_date],
                y=y2_range,
                mode="lines",
                line=dict(color="#02FF47", width=2),
                name="Current Date Line",
                yaxis="y2",
                showlegend=False
            )
            # X marker
            x_marker = go.Scatter(
                x=[curr_date],
                y=[predicted_values_val],
                mode="markers",
                marker=dict(color="#02FF47", size=18, symbol="x"),
                name="Current Date (predicted_values)",
                yaxis="y2",
                showlegend=False
            )
        else:
            # Placeholders if no current predicted_values (unlikely, but to keep consistent)
            curr_vline = go.Scatter(
                x=[],
                y=[],
                mode="lines",
                yaxis="y2",
                visible=False,
                showlegend=False
            )
            x_marker = go.Scatter(
                x=[],
                y=[],
                mode="markers",
                yaxis="y2",
                visible=False,
                showlegend=False
            )
        frame_data.append(curr_vline)
        frame_data.append(x_marker)

        frames.append(go.Frame(data=frame_data, name=str(curr_date.date())))


    init_data = frames[0].data
    axis_label_size = 18
    layout = go.Layout(
        title=dict(
            text=title,
            font=dict(color="white", size=16),  # Slightly larger font for readability
            x=0.5,      # Center the title horizontally
            xanchor='center'  # Anchor to center
        ),
        showlegend=False,
        xaxis=dict(
            title=dict(
                text="Date", 
                font=dict(color="white", size=axis_label_size),
                standoff=40  # Increase distance from tick labels to prevent overlap (adjust if needed)
            ),
            range=x_range,
            tickfont=dict(color="white", size=axis_label_size),
            gridcolor="#333333",  
            gridwidth=1,
            zerolinecolor="#333333"
        ),
        yaxis=dict(
            title=dict(text="Match (%)", font=dict(color="white", size=14)),
            side="left",
            range=y1_range,
            tickfont=dict(color="white", size=axis_label_size),
            gridcolor="#333333",  
            gridwidth=1,
            zerolinecolor="#333333"
        ),
        yaxis2=dict(
            title=dict(text=df_line_col, font=dict(color="white", size=14)),
            overlaying="y",
            side="right",
            showgrid=False,  
            range=y2_range,
            tickfont=dict(color="white", size=axis_label_size),
            zerolinecolor="#333333"
        ),
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        font=dict(color="white", size=12),  
        margin=dict(b=200),  # Expand bottom margin for space below the plot (adjust as needed)
        updatemenus=[{
            "type": "buttons",
            "direction": "left",  # Keeps buttons side-by-side horizontally
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 1000, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "quadratic-in-out"}
                        }
                    ]
                },
                {
                    "label": "Stop",
                    "method": "animate",
                    "args": [
                        [None],  # Stops at the current frame
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ],
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "x": 0.5,       # Center horizontally
            "xanchor": "center",  # Anchor to the center
            "y": -0.13,     # Position below x-axis labels (adjust if needed)
            "yanchor": "top",
            "font": dict(color="white")
        }],
        sliders=[dict(
            steps=[dict(method='animate',
                        args=[[f.name],
                              dict(mode='immediate',
                                   frame=dict(duration=frame_duration, redraw=True),
                                   transition=dict(duration=transition_duration))
                        ],
                        label=str(f.name)) for f in frames],
            active=0,
            font=dict(color="white"),
            bgcolor="#333333",
            currentvalue=dict(font=dict(color="white"), prefix="Date: "),
            x=0.5,          # Center horizontally
            xanchor="center",  # Anchor to the center
            y=-0.25,        # Move down below x-axis ticks/labels and buttons (adjust as needed)
            yanchor="top",
            len=1           # Full width; set to 0.9 for slightly shorter if preferred
        )]
    )
    
    
    fig = go.Figure(
        data=init_data,
        layout=layout,
        frames=frames
    )
    fig.update_layout(width=1024*1.5, height=1024/2)
    fig.show()
