from os import listdir
from os.path import dirname, join, isdir
from typing import List

from PIL import Image

from lib.functions import Sigmoid
from lib.layer import Layer
from lib.model import Model
from lib.training import DumbTrainer, Trainer

DATASET_TRAINING_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data/training_data")
DATASET_TESTING_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data/testing_data")

LETTERS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_SAMPLE_SIZE = (28, 28)
INPUT_SIZE = IMAGE_SAMPLE_SIZE[0] * IMAGE_SAMPLE_SIZE[1]
MIDDLE_SIZE = 32
OUTPUT_SIZE = len(LETTERS)


class ModelOCR:
    @staticmethod
    def create_and_train():
        letters = LETTERS[:10]
        limit = 2

        model = Model.load_or_create(Model(
            "model_ocr",
            Layer(INPUT_SIZE, MIDDLE_SIZE),
            Sigmoid(),
            Layer(MIDDLE_SIZE, OUTPUT_SIZE),
            Sigmoid(),
        ))
        # model.enable_debug()

        inputs, expected_outputs = ModelOCR._get_training_data(DATASET_TRAINING_PATH, limit=limit, letters=letters)
        test_inputs, test_outputs = ModelOCR._get_training_data(DATASET_TESTING_PATH, limit=limit, letters=letters)

        trainer = DumbTrainer(target_error=0.01, generations=200, generation_size=5, variation_factor=0.1)
        trainer.enable_debug()
        model = trainer.train(model, inputs, expected_outputs)

        outputs = model([
            ModelOCR._load_image("datasets/ocr/data/testing_data/0/28310.png"),
            ModelOCR._load_image("datasets/ocr/data/testing_data/1/28311.png"),
        ])
        print(ModelOCR._get_letters_from_outputs(outputs))

        print("Test dataset error:", Trainer.get_model_error(model, test_inputs, test_outputs))

        model.save()

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
            for index, val in enumerate(outputs_row):
                if val > _max[1]:
                    _max = (LETTERS[index], val)
            letters.append(_max)

        return letters

    @staticmethod
    def _get_training_data(path: str, limit: int, letters: List[str]):
        inputs = []
        expected_outputs = []
        paths = ModelOCR._get_training_paths(path, limit, letters)

        for letter, image_paths in paths.items():
            expected_outputs += [ModelOCR._get_letter_expected_output(letter)] * len(image_paths)
            inputs += list(map(ModelOCR._load_image, image_paths))

        return inputs, expected_outputs

    @staticmethod
    def _get_training_paths(path: str, limit: int, letters: List[str]):
        paths = {}
        items = listdir(path)

        for item in items:
            item_path = join(path, item)
            if isdir(item_path) and item in letters:
                item_sub_paths = listdir(item_path)[:limit]
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
