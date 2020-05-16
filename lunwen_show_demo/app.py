from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
import os
import tensorflow as tf
from config import SAVED_DIR, MODEL_PATH, HEIGHT, WIDTH, LABEL_MAP, DEFAULT_IMAGE_PATH
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'good good study, day day up'

model = tf.keras.models.load_model(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html', host_url=request.host_url, label='Bus',
                           image_path=os.path.join(request.host_url,
                                                   'static',
                                                   'images',
                                                   '1.jpg'))


@app.route('/classify', methods=['POST'])
def classify():
    image_ = request.files['image_file']
    if not image_.filename:
        save_path = DEFAULT_IMAGE_PATH
        filename = 'default.jpg'
    else:
        save_path = os.path.join(SAVED_DIR, image_.filename)
        image_.save(save_path)
        filename = image_.filename

    image = Image.open(save_path)
    image = image.resize((HEIGHT, WIDTH))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    result = model.predict(image)

    label = LABEL_MAP[result.argmax(axis=1)[0]]
    return render_template('index.html', host_url=request.host_url, label=label,
                           image_path=os.path.join(request.host_url, 'static', 'saved', filename))


if __name__ == '__main__':
    app.run()
