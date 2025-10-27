
import pickle
import dash
import os
import glob
import numpy
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go

from dash import Dash, dcc, html

from plotting import show_images, plot_objectives

PATH = "/Users/Anthony/Desktop/20251027_Multicolor-STED-RL"

def split_folders(folders):
    split_folders = []
    for folder in folders:
        split_folder = os.path.basename(folder)
        split_folders.append(split_folder)
    return split_folders

def create_app(path: str) -> Dash:

    global data
    global folders

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)

    folders = glob.glob(os.path.join(path, "*"))
    folders = [folder for folder in folders if os.path.isdir(folder)]
    folders.sort()

    data = pickle.load(open(f"{folders[-1]}/checkpoint.pkl", "rb"))

    app.layout = dbc.Container(
        [
            dcc.Interval(id="interval-component", interval=10 * 1000, n_intervals=0),
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        # dbc.Col([dcc.Clipboard(id="folder-copy", title="copy",)], width=1),
                        dbc.Col([dmc.MultiSelect(data=folders, value=[folders[-1]], id="folder", clearable=True, limit=100, searchable=True, nothingFound="No options found...", placeholder="Select model...", maxSelectedValues=1, style={"font-size" : 12}),], width=12)])
                ], width=11),
                dbc.Col([
                    html.I(className="fa fa-refresh", id="refresh-button", n_clicks=0, style={"background-color" : "#343434", "color" : "white", "border" : "none"})
                ], width=1, style={"text-align" : "center", "padding-right" : "0pt"}),
            ], style={"padding-top":"10pt", "padding-bottom":"0pt", "height":"5vh"}),
            dbc.Row([
                dbc.Col([
                    dcc.Slider(0, 30, 1, id="trial-slider", value=0, marks=None, tooltip={"placement": "left", "always_visible": False})
                ], width=12, style={"padding-top":"15pt", "height":"5vh"}),
            ]),
            dbc.Row([
                dbc.Row([
                    dcc.Graph(id="live-objectives", style={"height" : "100%"}),
                ]),
                dbc.Row([
                    dcc.Graph(id="live-images", style={"height" : "100%"}),
                ]),
            ], style={"height" : "90vh"}),
        ], style={"height" : "100vh", "background-color" : "#343434"}, fluid=True
    )

    @app.callback(
        dash.dependencies.Output("live-objectives", "figure"),
        [dash.dependencies.Input("interval-component", "n_intervals"),
         dash.dependencies.Input("folder", "value"),
         dash.dependencies.Input("trial-slider", "value")],
    )
    def update_objectives_callback(n, folder, trial):
        global data
        X = []
        y = []
        for info in data["info"]:
            if "actions" not in info:
                continue
            X.append(info["actions"])
            y.append(info["mo_objs"])
        X = numpy.array(X)
        y = numpy.array(y)

        config = {
            "obj_names" : ["Resolution", "Bleach", "SNR"],
        }
        y_ = {}
        for idx, obj_name in enumerate(config["obj_names"]):
            y_[obj_name] = y[:, :, idx]
        fig = plot_objectives(config, X, y_, _type="line", step=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=go.layout.Margin(
                l=10, #left margin
                r=10, #right margin
                b=50, #bottom margin
                t=50  #top margin
            )
        )
        fig.update_xaxes(
            color="white", showgrid=False
        )
        fig.update_yaxes(
            color="white", showgrid=False
        )

        return fig

    @app.callback(
        dash.dependencies.Output("live-images", "figure"),
        [dash.dependencies.Input("refresh-button", "n_clicks"),
         dash.dependencies.Input("folder", "value"),
         dash.dependencies.Input("trial-slider", "value")],
    )
    def update_images_callback(n, folder, trial):
        global data

        idx = trial + 1
        if idx >= len(data["info"]):
            idx = len(data["info"]) - 1
        info = data["info"][idx]

        fig = show_images(None, info["conf1s"], info["sted_images"], info["conf2s"])

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=go.layout.Margin(
                l=10, #left margin
                r=10, #right margin
                b=50, #bottom margin
                t=50  #top margin
            )
        )
        fig.update_xaxes(
            color="white", showgrid=False
        )
        fig.update_yaxes(
            color="white", showgrid=False
        )

        return fig

    @app.callback(
        dash.dependencies.Output("interval-component", "n_intervals"),
        [dash.dependencies.Input("interval-component", "n_intervals"),
         dash.dependencies.Input("folder", "value")],
    )
    def update_callback(n, folder):
        global data

        data = pickle.load(open(f"{folder[0]}/checkpoint.pkl", "rb"))

    return app

if __name__ == "__main__":
    
    app = create_app(PATH)
    app.run_server(debug=True)