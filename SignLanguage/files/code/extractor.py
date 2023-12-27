from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

        if weights is None:

            # Get model with pretrained weights.
            #input_tensor = Input(shape=(299, 299, 3))
            base_model = InceptionV3(
             #   input_tensor=input_tensor,
                weights='imagenet',
                include_top=True,
                input_shape=(299, 299, 3)
            )

            # We'll extract features at the final pool layer.
            self.model = Model(
                input=base_model.input,
                output=base_model.get_layer('avg_pool').output
            )

        else:
            # Load the model first.
            self.model = load_model(weights)

            # Then remove the top so we get features not predictions.
            # From: https://github.com/fchollet/keras/issues/2371
            self.model.layers.pop()
            self.model.layers.pop()  # two pops to get to pool layer
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        print("   here 1")
        print("   image path = ", image_path)
        img = image.load_img(image_path, target_size=(299, 299))
        print("   here 2")
        x = image.img_to_array(img)

        print("   here 3")
        x = np.expand_dims(x, axis=0)
        print("   here 4")

        x = preprocess_input(x)
        print("   here 5")
        print(x)
        # Get the prediction.
        features = self.model.predict(x)
        print("features = ", features)
        print("   here 6")
        if self.weights is None:
            # For imagenet/default network:
            features = features[0]#[0][0]
        else:
            # For loaded network:
            features = features[0]

        return features
