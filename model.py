import numpy as np
import tensorflow as tf

from tensorflow.keras.models import model_from_json


class FacialExpressionModel(object):

    """ A Class for Predicting the emotions using the pre-trained Model weights"""

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]


    def __init__(self, model_json_file, model_weights_file):

        # load model from JSON file which we created during Training
        with open(model_json_file, "r") as json_file:

            # Reading the json file and storing it in loaded_model
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # loading weights into the model
        self.loaded_model.load_weights(model_weights_file)

    def predict_emotion(self, img):
        """ It predicts the Emotion using our pre-trained model"""

        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

    def return_probabs(self, img):
        """  It returns the Probabilities of each emotions"""

        self.preds = self.loaded_model.predict(img)
        return self.preds
