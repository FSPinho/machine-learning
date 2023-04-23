from os import listdir
from os.path import dirname, join, isdir
from typing import List

from PIL import Image

from lib.functions import Sigmoid
from lib.layer import Layer
from lib.model import Model

DATASET_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data/training_data")
LETTERS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_SAMPLE_SIZE = (28, 28)


class ModelOCR:
    @staticmethod
    def create_and_train():
        model = Model.load_or_create(Model(
            "model_ocr",
            Layer(28 * 28, 16),
            Sigmoid(),
            Layer(16, len(LETTERS)),
            Sigmoid(),
        ))

        inputs, expected_outputs = ModelOCR._get_training_data()
        print(len(expected_outputs[0]))
        outputs = model([ModelOCR._load_image("datasets/ocr/data/testing_data/A/28320.png")])

        print(ModelOCR._get_letters_from_outputs(outputs))

        model.save()

    @staticmethod
    def _get_training_data():
        inputs = []
        expected_outputs = []
        paths = ModelOCR._get_training_paths()

        for letter, image_paths in paths.items():
            expected_outputs += [ModelOCR._get_letter_expected_output(letter)] * len(image_paths)
            inputs += list(map(ModelOCR._load_image, image_paths))

        return inputs, expected_outputs

    @staticmethod
    def _get_letter_expected_output(letter):
        output = [0.0] * len(LETTERS)
        output[ModelOCR._get_letter_index(letter)] = 1.0
        return output

    @staticmethod
    def _get_letter_index(letter: str):
        return LETTERS.index(letter)

    @staticmethod
    def _get_letters_from_outputs(outputs: List[List[int]]):
        letters = []

        for outputs_row in outputs:
            _max = ("_", 0.0)
            for index, val in outputs_row:
                if val > _max[1]:
                    _max = (LETTERS[index], val)

        return letters

    @staticmethod
    def _get_training_paths():
        paths = {}
        items = listdir(DATASET_PATH)

        for item in items:
            item_path = join(DATASET_PATH, item)
            if isdir(item_path) and item == "A":
                item_sub_paths = listdir(item_path)
                paths[item] = [join(item_path, path) for path in item_sub_paths if path.endswith(".png")]

        return paths

    @staticmethod
    def _load_image(path):
        image = Image.open(path)
        image.thumbnail(IMAGE_SAMPLE_SIZE)
        pixels = image.load()
        corner_pixel = pixels[0, 0]
        w, h = image.size

        resized_image = Image.new("L", IMAGE_SAMPLE_SIZE, corner_pixel)
        resized_pixels = resized_image.load()
        rw, rh = resized_image.size

        dw = (rw - w) // 2
        dh = (rh - h) // 2

        for x in range(w):
            for y in range(h):
                resized_pixels[x + dw, y + dh] = pixels[x, y]

        return [
            resized_pixels[x, y] / 255.0
            for y in range(rh)
            for x in range(rw)
        ]
