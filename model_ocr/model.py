import random
from os import listdir
from os.path import dirname, join, isdir, exists
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from PIL import Image as PImage, ImageDraw, ImageEnhance, ImageFont

from lib.functions import Accuracy, Normalize
from lib.training import Trainer
from lib.util.image import Image

MODELS_PATH = join(dirname(dirname(__file__)), "trained_models")
DATASET_TRAINING_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data2/training_data")
DATASET_TESTING_PATH = join(dirname(dirname(__file__)), "datasets/ocr/data2/testing_data")

LETTERS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_SIZE = 32
INPUT_SIZE = IMAGE_SIZE
OUTPUT_SIZE = len(LETTERS)


class ActivationHistoryMixin:
    def __init__(self):
        super().__init__()
        self._activations = []
        self._save_activations = False

    @property
    def activations(self):
        return self._activations

    def enable_save_activations(self):
        self._save_activations = True

    def _append_activation(self, x):
        if self._save_activations:
            x = Normalize()(x[0])
            if len(x.shape) == 3:
                self._activations.append([list(torch.flatten(c).tolist()) for c in x])
            else:
                self._activations.append([list(torch.flatten(x).tolist())])

    def _clear_activations(self):
        self._activations = []


class ModelCharClassifier(ActivationHistoryMixin, nn.Module):
    def __init__(self):
        super().__init__()

        # Input size = 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, 11, padding=4),  # 30x30
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 15x15
            # nn.Dropout2d(p=0.5),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 32, 6),  # 10x10
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 5x5
            # nn.Dropout2d(p=0.5),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(32 * 5 * 5, 400),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(400, OUTPUT_SIZE),
        )

    def forward(self, x):
        self._clear_activations()
        self._append_activation(x)

        layers = (
            self.conv1,
            self.conv2,
            self.fc1,
            self.fc2,
        )

        for layer in layers:
            x = layer(x)
            self._append_activation(x)

        return x


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = torch.relu(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step)
    )
    return loss


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step):
    loglikelihood = loglikelihood_loss(y, alpha)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return loglikelihood + kl_div


def loglikelihood_loss(y, alpha):
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def kl_divergence(alpha, num_classes):
    ones = torch.ones([1, num_classes], dtype=torch.float32)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


class ModelCharTrainer:
    @staticmethod
    def train_char_classifier():
        data_train = MNIST("./data/mnist",
                           download=True,
                           train=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
        print(data_train)
        return

        model = ModelCharClassifier()
        inputs, expected_outputs = ModelCharTrainer._get_training_data(
            path=DATASET_TRAINING_PATH,
            limit=10,
            letters=LETTERS,
        )

        ModelCharTrainer.train(model, inputs, expected_outputs, criterion=edl_mse_loss)

    @staticmethod
    def train(model: nn.Module, inputs, expected_outputs, criterion, epochs=200):
        print(f"Training {model.__class__.__name__} size={len(inputs)}")

        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
        save_points = 0
        chunk_size = 10
        accuracy = 0.0

        for epoch in range(epochs):
            model.train(True)
            torch.set_grad_enabled(True)

            inputs, expected_outputs = Trainer.shuffle_data(inputs, expected_outputs)
            chunks = Trainer.chunkify_data(inputs, expected_outputs, chunk_size)
            last_loss_values = []

            for i, (inputs_row, expected_outputs_row) in enumerate(chunks):
                optimizer.zero_grad()
                outputs_row = model(inputs_row)
                loss = criterion(outputs_row, expected_outputs_row, epoch, OUTPUT_SIZE, 100)
                loss.backward()
                optimizer.step()

                last_loss_values.append(loss.item())

            last_loss_avg = sum(last_loss_values) / len(last_loss_values)

            print(
                f"Epoch={epoch + 1}/{epochs} SavePts={save_points} " +
                f"Loss={last_loss_avg:.8f} " +
                f"Accuracy={accuracy:.2f}%",
                end="\r", flush=True
            )

            if (epoch + 1) % 5 == 0:
                save_points += 1
                ModelCharTrainer.save_model(model)

            if epoch % 10 == 0:
                model.train(False)
                torch.set_grad_enabled(False)
                accuracy = Accuracy()(model(inputs), expected_outputs)

        print("")
        print("\nTraining finished.")

    @staticmethod
    def get_model(model_cls):
        path = ModelCharTrainer.get_persistence_path(model_cls.__name__)
        model = model_cls()
        if exists(path):
            model.load_state_dict(torch.load(path))
            model.eval()
            return model
        return model

    @staticmethod
    def save_model(model):
        torch.save(model.state_dict(), ModelCharTrainer.get_persistence_path(model.__class__.__name__))

    @staticmethod
    def get_persistence_path(key):
        return join(MODELS_PATH, f"{key}.data")

    @staticmethod
    def scan_image():
        model = ModelCharTrainer.get_model(ModelCharClassifier)

        # image_path = "/Users/felipepinho/Downloads/ocr_tests/1_1.jpeg"
        # image_path = "/Users/felipepinho/Downloads/ocr_tests/3_1.png"
        # image_path = "/Users/felipepinho/Downloads/ocr_tests/numbers_2.jpeg"
        image_path = "/Users/felipepinho/Downloads/ocr_tests/letters_1.png"
        # image_path = "/Users/felipepinho/Downloads/ocr_tests/letters_2.png"
        image_origin = Image.get_image(image_path)

        image = image_origin.convert("L")
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        image_out = image.copy()
        image_out_draw = ImageDraw.Draw(image_out)
        w, h = image.size

        scan_sizes = [100 + i * 4 for i in range(0, 4)]
        print(scan_sizes)

        findings = []
        target_confidence = 1.0 - 1e-10

        font = ImageFont.truetype(join(dirname(dirname(__file__)), "fonts/Roboto-Medium.ttf"), size=32)

        for size in scan_sizes:
            if size <= w and size <= h:
                base_scaled_w = w * IMAGE_SIZE / size
                base_scaled_h = h * IMAGE_SIZE / size
                base_scaled_image = image.copy()
                base_scaled_image.thumbnail((base_scaled_w, base_scaled_h))

                scan_move_step = 2
                w_steps = int((base_scaled_w - IMAGE_SIZE) // scan_move_step)
                h_steps = int((base_scaled_h - IMAGE_SIZE) // scan_move_step)

                print(f"Scanning size={size} step_size={scan_move_step} step_matrix={w_steps}x{h_steps}")

                for y in range(-1, h_steps + 1):
                    for x in range(-1, w_steps + 1):
                        rect = (
                            int(x * scan_move_step),
                            int(y * scan_move_step),
                            int(x * scan_move_step + IMAGE_SIZE),
                            int(y * scan_move_step + IMAGE_SIZE)
                        )

                        tmp_image = base_scaled_image.crop(rect)

                        inputs = torch.stack([Image.get_image_tensors(tmp_image)])
                        outputs_classifier = model(inputs)
                        letter, confidence = tuple(ModelCharTrainer.get_letters_from_tensor(outputs_classifier)[0])
                        findings.append((inputs, letter, confidence, size))

                        if confidence >= target_confidence:
                            rect = tuple(map(lambda _x: _x * size / IMAGE_SIZE, rect))
                            image_out_draw.rectangle(rect, outline=0)
                            image_out_draw.text(rect, letter, font=font)

        for inputs, letter, confidence, size in findings:
            if confidence >= target_confidence:
                print(f"Recognized letter: letter={letter} " +
                      f"confidence={confidence:.8f} size={size}x{size}")
                Image.print_image(inputs[0])

        out_path = image_path.split(".")[0] + "-out.png"
        image_out.save(out_path, format="PNG")

    @staticmethod
    def show_tests():
        model = ModelCharTrainer.get_model(ModelCharClassifier)
        test_inputs, _ = ModelCharTrainer._get_training_data(DATASET_TESTING_PATH, 1, LETTERS)

        print(f"Inputs length={len(test_inputs)}")

        for test_inputs_row in test_inputs:
            test_outputs_row = model(torch.stack([test_inputs_row]))

            matched_letters = ModelCharTrainer.get_letters_from_tensor(test_outputs_row)
            print(f"Recognized letter={matched_letters[0][0]} confidence={matched_letters[0][1]:.8f}")
            Image.print_image(test_inputs_row)
        print(LETTERS)

    @staticmethod
    def _get_letter_tensors(letter=None):
        output = [0.0] * len(LETTERS)
        if letter:
            output[ModelCharTrainer._get_letter_index(letter)] = 1.0
        return torch.tensor(output)

    @staticmethod
    def _get_letter_index(letter: str):
        return LETTERS.index(letter)

    @staticmethod
    def get_letters_from_tensor(outputs: torch.Tensor):
        letters = []

        for outputs_row in outputs:
            _max = ("_", float('-inf'))
            for index, val in enumerate(outputs_row):
                if val > _max[1]:
                    _max = (LETTERS[index], val)
            letters.append(_max)

        return letters

    @staticmethod
    def _get_training_data(path: str, limit: int, letters: List[str]):
        inputs = []
        expected_outputs = []
        paths = ModelCharTrainer._get_training_paths(path, limit, letters)
        items = sorted(paths.items(), key=lambda x: x[0])

        for letter, image_paths in items:
            for path in image_paths:
                image = Image.get_image(path)
                image = Image.get_scales_image(image, IMAGE_SIZE)
                image_tensors = Image.get_image_tensors(image)
                inputs.append(image_tensors)
                expected_outputs.append(ModelCharTrainer._get_letter_tensors(letter=letter))

                # variations = ModelCharTrainer._generate_image_variations(image)
                # for variation in variations:
                #     inputs.append(Image.get_image_tensors(variation))
                #     expected_outputs.append(ModelCharTrainer._get_letter_tensors())

        # variations = (
        #         ModelCharTrainer._generate_shapes_image_variations()
        #         + ModelCharTrainer._generate_blank_image_variations()
        #         + ModelCharTrainer._generate_random_image_variations()
        # )
        # for variation in variations:
        #     inputs.append(Image.get_image_tensors(variation))
        #     expected_outputs.append(ModelCharTrainer._get_letter_tensors())

        return torch.stack(inputs), torch.stack(expected_outputs)

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
    def _generate_image_variations(image: PImage.Image):
        pixels = image.load()
        w, h = image.size
        variations = []
        variations_templates = [
            ("   "
             "###"
             "   ", 3),

            (" # "
             " # "
             " # ", 3),

            ("## "
             " # "
             " ##", 3),

            (" ##"
             " # "
             "## ", 3),

            ("###"
             "## "
             "#  ", 3),

            ("  #"
             " ##"
             "###", 3),

            ("###"
             " ##"
             "  #", 3),

            ("#  "
             "## "
             "###", 3),

            ("###"
             "# #"
             "###", 3),

            ("    "
             "    "
             "####"
             "####", 4),

            ("####"
             "####"
             "    "
             "    ", 4),

            ("## "
             "## "
             "## ", 3),

            (" ##"
             " ##"
             " ##", 3),
        ]

        for template, template_size in variations_templates:
            variation = image.copy()
            draw = ImageDraw.Draw(variation)
            for i, char in enumerate(template):
                if char == "#":
                    x = (i % template_size) * w / template_size
                    y = (i // template_size) * h / template_size
                    draw.rectangle(
                        (x, y, x + w / template_size, y + h / template_size),
                        fill=pixels[0, 0]
                    )
            variations.append(variation)

        return variations

    @staticmethod
    def _generate_random_image_variations():
        pixel_probabilities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        variations = []

        for probability in pixel_probabilities:
            variation = Image.generate_image("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
            variation_pixels = variation.load()

            for i in range(IMAGE_SIZE * IMAGE_SIZE):
                y = i // IMAGE_SIZE
                x = i % IMAGE_SIZE
                p = random.uniform(0.0, 1.0) < probability
                variation_pixels[x, y] = 0 if p else 255

            variations.append(variation)

        return variations

    @staticmethod
    def _generate_shapes_image_variations():
        base = Image.generate_image("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
        variations = []

        # Rects
        for i in range(10):
            variation = base.copy()
            variations.append(variation)
            for j in range(i + 1):
                x = random.uniform(0, IMAGE_SIZE)
                y = random.uniform(0, IMAGE_SIZE)
                size = IMAGE_SIZE / 8
                draw = ImageDraw.Draw(variation)
                draw.rectangle((x, y, x + size, y + size), fill=0)

        # Circles
        for i in range(10):
            variation = base.copy()
            variations.append(variation)
            for j in range(i + 1):
                x = random.uniform(0, IMAGE_SIZE)
                y = random.uniform(0, IMAGE_SIZE)
                size = IMAGE_SIZE / 8
                draw = ImageDraw.Draw(variation)
                draw.ellipse((x, y, x + size, y + size), fill=0)

        return variations

    @staticmethod
    def _generate_blank_image_variations():
        return [
            Image.generate_image("L", (IMAGE_SIZE, IMAGE_SIZE), i)
            for i in [0, 50, 100, 150, 200, 255]
        ]
