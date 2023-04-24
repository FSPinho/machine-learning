from os import listdir
from os.path import dirname, join, isdir, exists
from typing import List

import torch as t
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageEnhance

from lib.training import Trainer

MODELS_PATH = join(dirname(dirname(__file__)), "trained_models")
DATASET_TRAINING_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data/training_data")
DATASET_TESTING_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data/testing_data")

LETTERS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:10]
IMAGE_SAMPLE_SIZE = (24, 24)
INPUT_SIZE = IMAGE_SAMPLE_SIZE[0] * IMAGE_SAMPLE_SIZE[1]
HIDDEN_1_SIZE = 32
HIDDEN_2_SIZE = 32
OUTPUT_SIZE = len(LETTERS)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(INPUT_SIZE, HIDDEN_1_SIZE)
        self.input_layer_act = nn.ReLU()
        self.hidden_layer_1 = nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE)
        self.hidden_layer_1_act = nn.ReLU()
        self.hidden_layer_2 = nn.Linear(HIDDEN_2_SIZE, OUTPUT_SIZE)
        self.hidden_layer_2_act = nn.Sigmoid()

    def forward(self, x):
        steps = [
            self.input_layer,
            self.input_layer_act,
            self.hidden_layer_1,
            self.hidden_layer_1_act,
            self.hidden_layer_2,
            self.hidden_layer_2_act
        ]
        for step in steps:
            x = step(x)
        return x


class ModelOCR:
    @staticmethod
    def get_model():
        path = ModelOCR.get_persistence_path()
        model = Model()
        if exists(path):
            model.load_state_dict(t.load(path))
            model.eval()
            return model
        return model

    @staticmethod
    def save_model(model):
        t.save(model.state_dict(), ModelOCR.get_persistence_path())

    @staticmethod
    def get_persistence_path():
        return join(MODELS_PATH, f"model_ocr.data")

    @staticmethod
    def scan_image():
        model = ModelOCR.get_model()

        image_path = "/Users/felipepinho/Downloads/ocr_tests/3_1.png"
        image_path = "/Users/felipepinho/Downloads/ocr_tests/numbers_3.jpeg"
        image = Image.open(image_path)
        w, h = image.size

        scan_sizes = [int(128 * (1 + i * 0.25)) for i in range(0, 40)]
        print(scan_sizes)

        findings = []

        for size in scan_sizes:
            if size <= w and size <= h:
                scan_move_step = size / 8
                w_steps = int((w - size) // scan_move_step)
                h_steps = int((h - size) // scan_move_step)

                print(f"Scanning size={size} step_size={scan_move_step:.4f}")

                for y in range(0, h_steps):
                    for x in range(0, w_steps):
                        rect = (
                            x * scan_move_step,
                            y * scan_move_step,
                            x * scan_move_step + size,
                            y * scan_move_step + size
                        )
                        tmp_image = image.crop(rect).convert("L")
                        enhancer = ImageEnhance.Contrast(tmp_image)
                        tmp_image = enhancer.enhance(1.0)
                        tmp_path = "/tmp/tmp-image.png"
                        tmp_image.save(tmp_path)
                        inputs = t.tensor([ModelOCR._load_image(tmp_path)])
                        findings.append((inputs, model(inputs), size))

        for inputs, outputs, size in findings:
            letter = ModelOCR._get_letters_from_outputs(outputs)
            if letter[0][1] > 0.99999:
                print("Recognized letter:", letter, f"{size}x{size}")
                ModelOCR._print_image_inputs(inputs[0])

    @staticmethod
    def show_tests():
        model = ModelOCR.get_model()
        test_inputs, _ = ModelOCR._get_training_data(DATASET_TESTING_PATH, 20, LETTERS)

        for i in range(len(test_inputs)):
            test_inputs_row = test_inputs[i:i + 1]
            test_outputs_row = model(test_inputs_row)
            matched_letters = ModelOCR._get_letters_from_outputs(test_outputs_row)
            print("Recognized letter:", matched_letters)
            ModelOCR._print_image_inputs(test_inputs_row[0])

    @staticmethod
    def create_and_train():
        model = ModelOCR.get_model()
        inputs, expected_outputs = ModelOCR._get_training_data(DATASET_TRAINING_PATH, 500, LETTERS[:10])
        test_inputs, test_expected_outputs = ModelOCR._get_training_data(DATASET_TESTING_PATH, 500, LETTERS[:10])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        print(f"Training size={len(inputs)}")

        for epoch in range(10000):
            inputs, expected_outputs = Trainer.shuffle_data(inputs, expected_outputs)

            chunk_size = 10
            chunks = Trainer.chunkify_data(inputs, expected_outputs, chunk_size)
            loss = 0.0
            loss_count = 0.0

            for i, (inputs_row, expected_outputs_row) in enumerate(chunks):
                optimizer.zero_grad()
                outputs_row = model(inputs_row)
                loss = criterion(outputs_row, expected_outputs_row)
                loss.backward()
                optimizer.step()

                loss += loss.item()
                loss_count += 1

            if epoch % 100 == 0:
                ModelOCR.save_model(model)

            error = Trainer.get_model_error(model, inputs, expected_outputs)
            testing_error = Trainer.get_model_error(model, test_inputs, test_expected_outputs)

            print(
                f"Epoch={epoch + 1} Loss={loss / loss_count:.8f} Error={error:.8f} TError={testing_error:.8f}",
                end="\r", flush=True
            )

        print("")
        print("\nTraining finished.")

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

        return t.tensor(inputs), t.tensor(expected_outputs)

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
        div = 2

        for i in range(h // div):
            line = ""
            for j in range(w // div):
                pixel = inputs[div * i * w + div * j] * 0.9999
                pixel_grey_scale = grey_scale[int(pixel * len(grey_scale))]
                line += pixel_grey_scale
            output_lines.append(line)

        print("\n".join(output_lines))
