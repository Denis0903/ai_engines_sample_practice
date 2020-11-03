from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import sys
 
# Input Images From Command Line
input_filename = sys.argv[1]
input_image = image.load_img(input_filename, target_size=(224, 224))
input = np.expand_dims(image.img_to_array(input_image), axis=0)
 
# Define Models
model = VGG16(weights='imagenet')
results = model.predict(preprocess_input(input))
 
# Decoding Predictions
decoded_results = decode_predictions(results, top=5)[0]
for decoded_result in decoded_results:
    print(decoded_result)