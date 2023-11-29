import numpy as np
import tflite_runtime.interpreter as tflite
from urllib import request
from PIL import Image
from io import BytesIO

interpreter = tflite.Interpreter(model_path='bees-wasps-v2.tflite')
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    prepared_image = prepare_image(img, (150, 150))
    X = np.array(prepared_image) * (1. / 255)
    X = np.array([X], dtype="float32")

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    return preds.tolist()

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result