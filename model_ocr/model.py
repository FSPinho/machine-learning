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
IMAGE_SAMPLE_SIZE = (16, 16)
INPUT_SIZE = IMAGE_SAMPLE_SIZE[0] * IMAGE_SAMPLE_SIZE[1]
MIDDLE_SIZE = 16
OUTPUT_SIZE = len(LETTERS)


class ModelOCR:
    @staticmethod
    def scan_image():
        model = Model.load("model_ocr")

        # image_path = "datasets/ocr/data/testing_data/0/28310.png"
        image_path = "datasets/ocr/data/testing_data/1/28311.png"
        image = Image.open(image_path)
        w, h = image.size

        scan_sizes = [size for size in range(512, 14, -2)]
        scan_move_step = 1

        findings = []

        for size in scan_sizes:
            if size <= w and size <= h:
                for x in range(0, (w - size) // scan_move_step):
                    for y in range(0, (h - size) // scan_move_step):
                        rect = (x, y, x + size, y + size)
                        image = image.crop(rect).convert("L")
                        tmp_path = "/tmp/tmp-image.png"
                        image.save(tmp_path)
                        inputs = [ModelOCR._load_image(tmp_path)]
                        findings.append((inputs, model(inputs), size))

        for inputs, outputs, size in findings:
            letter = ModelOCR._get_letters_from_outputs(outputs)
            if letter[0][1] > 0.7:
                print("Recognized letter:", letter, f"{size}x{size}")
                ModelOCR._print_image_inputs(inputs[0])

    @staticmethod
    def show_tests():
        model = Model.load("model_ocr")
        test_inputs, _ = ModelOCR._get_training_data(DATASET_TESTING_PATH, 2, LETTERS[:11])

        for test in test_inputs[:20]:
            print("Recognized letter:", ModelOCR._get_letters_from_outputs(model([test])))
            ModelOCR._print_image_inputs(test)

    @staticmethod
    def create_and_train():
        letters = LETTERS[:10]
        limit = 1000

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

        trainer = DumbTrainer(target_error=0.01, generations=100, generation_size=4, variation_factors=[0.005])
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

    @staticmethod
    def _print_image_inputs(inputs: List[float]):
        grey_scale = " .:-=+*#%@"
        output_lines = []
        w, h = IMAGE_SAMPLE_SIZE
        div = 1

        for i in range(h // div):
            line = ""
            for j in range(w // div):
                pixel = inputs[div * i * w + div * j]
                pixel_grey_scale = grey_scale[int(pixel * len(grey_scale))]
                line += pixel_grey_scale
            output_lines.append(line)

        print("\n".join(output_lines))
