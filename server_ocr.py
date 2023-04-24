from flask import Flask

app = Flask("ocr")


@app.route("/ocr")
def hello_world():
    return "<p>Hello, World!</p>"
