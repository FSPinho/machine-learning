from json import dumps

import torch
from flask import Flask
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/ocr/<data>")
@cross_origin()
def hello_world(data):
    # noinspection PyBroadException
    try:
        from model_ocr.model import ModelCharClassifier, ModelCharTrainer, IMAGE_SIZE

        data = list(map(float, data.split("_")))
        data = [[data[i: i + IMAGE_SIZE] for i in range(0, len(data), IMAGE_SIZE)]]

        model = ModelCharTrainer.get_model(ModelCharClassifier)
        model.enable_save_activations()
        model.eval()

        inputs = torch.tensor([data])

        outputs = model(inputs)
        activations = model.activations

        guess = ModelCharTrainer.get_letters_from_tensor(outputs)

        confidence = float(guess[0][1])
        letter = guess[0][0]

        return dumps(dict(letter=letter, confidence=confidence, activations=activations))

    except Exception as e:
        print(e)
        return dumps(dict(letter="_", confidence=0.0, activations=[]))
