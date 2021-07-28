from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

from flask import Flask

app = Flask(__name__)

# loding the model
model = load_model('models/apple2.h5')

def model_predict(img_path, model):

  test_image = image.load_img(img_path, target_size=(224, 224))
  test_image = image.img_to_array(test_image)
  test_image = test_image/255
  test_image = np.expand_dims(test_image, axis=0)
  result     = model.predict(test_image)

  return result


@app.route('/', methods=['GET'])
def index():
  return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    # get the file from post request
    f = request.files['file']

    # save the file to uploads folder
    basepath = os.path.dirname(os.path.realpath('__file__'))
    file_path = os.path.join(basepath,'uploads', secure_filename(f.filename)) 
    f.save(file_path)

    # Making predictions
    result = model_predict(file_path, model)

    # there are 4 categories
    categories = ['Healthy', 'Multiple Disease', 'Rust', 'Scab']

    # process your result for humans
    # Returns the indices of the maximum values along an axis.
    pred_class = result.argmax()

    output = categories[pred_class]
    return output
  else:
    return render_template('test.html')

  return None # if request method is not POST then return None


if __name__ == '__main__':
  app.run(debug=True)
