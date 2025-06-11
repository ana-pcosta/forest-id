import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import numpy as np
import torch  # Assuming PyTorch model
import base64
import io
from PIL import Image
from dotenv import load_dotenv
import json
from forestid.modelhandler import ModelHandler
from forestid.datasethandler import PlantDataset
from forestid import ROOT_PATH

load_dotenv(".env")

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("🌿 Plant Image Classifier", className="text-center mt-3"),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id="upload-image",
                    children=dbc.Button(
                        "📂 Upload Image", color="primary", className="mt-3"
                    ),
                    multiple=False,
                    style={"textAlign": "center"},
                ),
                width=12,
                className="text-center",
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Spinner(
                    size="lg",
                    color="primary",
                    children=[
                        html.Div(id="output-image-upload", className="text-center mt-4")
                    ],
                ),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                html.H3(id="prediction-output", className="text-center mt-4"),
                width=12,
            )
        ),
    ],
    fluid=True,
)


def parse_image(contents):
    """Decode the uploaded image and preprocess it."""
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    return image


@app.callback(
    [
        Output("output-image-upload", "children"),
        Output("prediction-output", "children"),
    ],
    Input("upload-image", "contents"),
    prevent_initial_call=True,
)
def update_output(contents):
    if contents is not None:
        # Process the image
        with open(ROOT_PATH + "/model/class_mapping.json", "r") as f:
            class_map = json.load(f)

        dataset = PlantDataset(
            [], [], output_size=(224, 224), class_id_mapping=class_map
        )
        image = parse_image(contents)
        img_tensor = dataset.transform(image).unsqueeze(0)  # Add batch dimension
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_tensor = img_tensor.to(device)

        # Run inference
        with torch.no_grad():
            modelhandler = ModelHandler()
            model = modelhandler.load_model(
                ROOT_PATH + "/model/baseline_best_model.pth", len(class_map)
            )
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            top_prob, top_class = torch.max(probs, dim=0)

        # Get predicted class label
        predicted_label = dataset.get_class_name(top_class.item())
        confidence = top_prob.item() * 100

        # Display uploaded image and prediction
        return [
            html.Img(
                src=contents,
                style={
                    "width": "300px",
                    "borderRadius": "10px",
                    "boxShadow": "0px 4px 10px rgba(0,0,0,0.1)",
                },
            ),
            dbc.Alert(
                f"Prediction: {predicted_label} ({confidence:.2f}%)",
                color="success",
                className="text-center mt-3",
            ),
        ]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
