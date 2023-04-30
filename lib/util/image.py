from typing import Union

import torch
from PIL import Image as PImage


class Image:
    @staticmethod
    def generate_image(mode, size, color):
        return PImage.new(mode, size, color)

    @staticmethod
    def get_image(path: str) -> PImage.Image:
        return PImage.open(path)

    @staticmethod
    def get_image_tensors(image: PImage.Image):
        pixels = image.load()
        w, h = image.size

        def f(x):
            return x / 256.0

        return torch.tensor([[
            [
                f(pixels[j, i])
                for j in range(w)
            ]
            for i in range(h)
        ]])

    @staticmethod
    def get_scales_image(image: PImage.Image, target_size: int):
        image.thumbnail((target_size, target_size))
        pixels = image.load()
        w, h = image.size

        if image.size[0] == target_size and image.size[1] == target_size:
            return image.copy()

        scaled_image = PImage.new("L", (target_size, target_size), pixels[0, 0])
        scaled_pixels = scaled_image.load()

        h_offset = (target_size - w) // 2
        v_offset = (target_size - h) // 2

        for x in range(w):
            for y in range(h):
                scaled_pixels[x + h_offset, y + v_offset] = pixels[x, y]

        return scaled_image

    @staticmethod
    def print_image(image: Union[PImage.Image, torch.Tensor]):
        if isinstance(image, PImage.Image):
            image = Image.get_image_tensors(image)

        data = image[0] if len(image.shape) == 3 else image
        grey_scale = " .:-=+*#%@"

        output_lines = []
        scale = 0.5

        for y in range(int(len(data) * scale)):
            line = ""
            for x in range(int(len(data[0]) * scale)):
                pixel = data[int(y / scale), int(x / scale)]
                pixel_grey_scale = grey_scale[int(float(pixel) * len(grey_scale))]
                line += pixel_grey_scale
            output_lines.append(line)

        print("\n".join(output_lines))