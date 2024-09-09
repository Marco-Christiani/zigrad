import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

app = dash.Dash(__name__)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.cdnfonts.com/css/d-din" rel="stylesheet">
        <style>
            body {
                font-family: 'D-DIN', sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    style={"backgroundColor": "#1E1E1E", "color": "#F0F0F0"},
    children=[
        html.H1(
            "DQN Metrics Dashboard", style={"textAlign": "center", "color": "#F0F0F0"}
        ),
        html.Div(
            [
                html.Label("Smoothing Window:", style={"color": "#F0F0F0"}),
                dcc.Slider(
                    id="smoothing-slider",
                    min=50,
                    max=1000,
                    step=50,
                    value=100,
                    marks={i: str(i) for i in range(50, 1000, 150)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"width": "50%", "padding": "20px", "margin": "auto"},
        ),
        html.Button(
            "Update Now",
            id="manual-update-button",
            style={"marginTop": "20px", "backgroundColor": "#4CAF50", "color": "white"},
        ),
        dcc.Checklist(
            id="auto-update-toggle",
            options=[{"label": "Enable Auto Update", "value": "enabled"}],
            value=[],
            style={"marginTop": "10px", "color": "#F0F0F0"},
        ),
        dcc.Graph(id="loss-graph"),
        dcc.Graph(id="total-reward-graph"),
        dcc.Graph(id="avg-action-graph"),
        # interval for auto-update
        dcc.Interval(
            id="interval-component", interval=1000, n_intervals=0, disabled=True
        ),
    ],
)


@app.callback(
    Output("interval-component", "disabled"), Input("auto-update-toggle", "value")
)
def toggle_auto_update(value):
    return "enabled" not in value


@app.callback(
    [
        Output("loss-graph", "figure"),
        Output("total-reward-graph", "figure"),
        Output("avg-action-graph", "figure"),
    ],
    [
        Input("manual-update-button", "n_clicks"),
        Input("interval-component", "n_intervals"),
        Input("smoothing-slider", "value"),
    ],
    [
        State("loss-graph", "relayoutData"),
        State("total-reward-graph", "relayoutData"),
        State("avg-action-graph", "relayoutData"),
    ],
)
def update_graphs(
    n_clicks,
    n_intervals,
    smoothing_window,
    relayout_loss,
    relayout_reward,
    relayout_action,
):
    df = pd.read_csv("/tmp/dqnlog.csv")

    def create_figure(df, x, y, smoothing_window, relayout_data):
        fig = px.scatter(
            df,
            x=x,
            y=y,
            trendline="rolling",
            trendline_options=dict(window=smoothing_window),
        )
        fig.update_layout(height=400, template="plotly_dark")
        fig.update_traces(marker_size=2, marker_opacity=0.6)

        if relayout_data:
            # Safely extract zoom and pan information
            x_range = [
                relayout_data.get("xaxis.range[0]", None),
                relayout_data.get("xaxis.range[1]", None),
            ]
            y_range = [
                relayout_data.get("yaxis.range[0]", None),
                relayout_data.get("yaxis.range[1]", None),
            ]

            # only update if we have valid ranges
            if all(x_range):
                fig.update_xaxes(range=x_range)
            if all(y_range):
                fig.update_yaxes(range=y_range)

        return fig

    fig_loss = create_figure(df, "episode", "loss", smoothing_window, relayout_loss)
    fig_total_reward = create_figure(
        df, "episode", "total_reward", smoothing_window, relayout_reward
    )
    fig_avg_action = create_figure(
        df, "episode", "avg_action", smoothing_window, relayout_action
    )

    return fig_loss, fig_total_reward, fig_avg_action


if __name__ == "__main__":
    app.run_server(debug=True)
