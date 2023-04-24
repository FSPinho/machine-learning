import sys

from model_ocr.model_2 import ModelOCR
from model_xor.model import ModelXOR


def main():
    if len(sys.argv) >= 3:
        _model_name = sys.argv[1]
        _model_command = sys.argv[2]

        if _model_name == "ocr":
            if _model_command == "train":
                ModelOCR.create_and_train()
            elif _model_command == "show":
                ModelOCR.show_tests()
            elif _model_command == "scan":
                ModelOCR.scan_image()

        elif _model_name == "xor":
            if _model_command == "train":
                ModelXOR.create_and_train()


if __name__ == '__main__':
    main()
