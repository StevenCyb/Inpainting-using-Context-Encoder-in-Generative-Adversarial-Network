import os
import cv2
import argparse
import numpy as np
from gan_ce.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image.")
ap.add_argument("-m", "--mask", required=True, help="Path to the mask.")
ap.add_argument("-o", "--output", required=True, help="Path to save prediction.")
ap.add_argument("-t", "--tiles", required=False, default=(2, 2), help="How many tiles should the picture be divided into (default=(2,2))).")
ap.add_argument("-sh", "--shape", required=False, default=(256, 256, 3), help="Define the shape of a tile (default=(256,256,3))).")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path to the weights (default='./weights/weights.ckpt').")
args = vars(ap.parse_args())

# Verify the passed parameters
if not isinstance(args["tiles"], tuple) or len(args["tiles"]) != 2:
    raise Exception("Tiles parameter is invalid. Should be something like '(2,2)'.")
if not isinstance(args["shape"], tuple) or len(args["shape"]) != 3:
    raise Exception("Shape parameter is invalid. Should be something like '(256,256,3)'.")
if not os.path.isdir(os.path.dirname(args["weights"])):
    raise Exception("Path to weights is invalid.")
if not os.path.isfile(args["image"]):
    raise Exception("Path to image is invalid.")
if not os.path.isfile(args["mask"]):
    raise Exception("Path to mask is invalid.")

# Load the image to inpaint
image = cv2.cvtColor(cv2.imread(args["image"], 3), cv2.COLOR_BGR2RGB)
# Load the mask to inpaint and norm it to [0,0,0] -> [1,1,1]
mask = cv2.cvtColor(cv2.imread(args["mask"], 3), cv2.COLOR_BGR2RGB)
mask[np.where((mask != [0, 0, 0]).all(axis=2))] = [1, 1, 1]

# Initalize the GAN (Context Encoder(Generator) and Discriminator) 
network = Network(tiles=args["tiles"], shape=args["shape"])
# Load the weights
network.load_weights(weights_path=args["weights"])
# Start prediction and save the results
prediction = network.predict(image, mask)
cv2.imwrite(args["output"], cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
