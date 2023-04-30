import sys

from model_ocr.model import ModelCharTrainer


def main():
    if len(sys.argv) >= 3:
        _model_name = sys.argv[1]
        _model_command = sys.argv[2]

        if _model_name == "ocr":
            model = ModelCharTrainer
            if _model_command == "train":
                model.train_char_classifier()
            elif _model_command == "show":
                model.show_tests()
            elif _model_command == "scan":
                model.scan_image()


if __name__ == '__main__':
    main()
