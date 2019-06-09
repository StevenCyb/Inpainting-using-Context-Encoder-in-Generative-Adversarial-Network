import os
import cv2
import argparse
from gan_ce.network import Network

# Define arguments with there default values
ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path", required=False, default='./training', help="Path to the training dataset (default='./training').")
ap.add_argument("-t", "--tiles", required=False, type=tuple, default=(2, 2), help="How many tiles should the picture be divided into (default=(2,2))).")
ap.add_argument("-sh", "--shape", required=False, type=tuple, default=(256, 256, 3), help="Define the shape of a tile (default=(256,256,3))).")
ap.add_argument("-e", "--epochs", required=False, type=int, default=50000, help="No. of training epochs (default=10000).")
ap.add_argument("-bs", "--batch_size", required=False, type=int, default=4, help="The batch size (default=4).")
ap.add_argument("-w", "--weights", required=False, default='./weights/weights.ckpt', help="Path where to store the weights (default='./weights/weights.ckpt').")
ap.add_argument("-se", "--saving_epochs", required=False, type=int, default=100, help="In which steps should the weights be stored (default=100).")
ap.add_argument("-mir", "--mask_min_rectangles", required=False, type=int, default=1, help="Min. ammount of rectangles for random mask (default=1).")
ap.add_argument("-mar", "--mask_max_rectangles", required=False, type=int, default=3, help="Max. ammount of rectangles for random mask (default=3).")
ap.add_argument("-mil", "--mask_min_lines", required=False, type=int, default=1, help="Min. ammount of lines for random mask (default=1).")
ap.add_argument("-mal", "--mask_max_lines", required=False, type=int, default=3, help="Max. ammount of lines for random mask (default=3).")
ap.add_argument("-mic", "--mask_min_circles", required=False, type=int, default=1, help="Min. ammount of circles for random mask (default=1).")
ap.add_argument("-mac", "--mask_max_circles", required=False, type=int, default=3, help="Max. ammount of circles for random mask (default=3).")
args = vars(ap.parse_args())

# Verify the passed parameters
if not os.path.isdir(args["dataset_path"]):
    raise Exception("Path to training dataset is invalid.")
if not isinstance(args["tiles"], tuple) or len(args["tiles"]) != 2:
    raise Exception("Tiles parameter is invalid. Should be something like '(2,2)'.")
if not isinstance(args["shape"], tuple) or len(args["shape"]) != 3:
    raise Exception("Shape parameter is invalid. Should be something like '(256,256,3)'.")
if not isinstance(args["epochs"], int) or args["epochs"] < 1:
    raise Exception("Epochs has an invalid value.")
if not isinstance(args["batch_size"], int) or args["batch_size"] < 1:
    raise Exception("Batch size has an invalid value.")
if not os.path.isdir(os.path.dirname(args["weights"])):
    raise Exception("Path to store weights is invalid.")
if not isinstance(args["saving_epochs"], int) or args["saving_epochs"] < 1:
    raise Exception("Saving epochs has an invalid value.")
if not isinstance(args["mask_min_rectangles"], int) or args["mask_min_rectangles"] < 0:
    raise Exception("Min. rectangle ammount has an invalid value.")
if not isinstance(args["mask_max_rectangles"], int) or args["mask_max_rectangles"] < 0:
    raise Exception("Max. rectangle ammount has an invalid value.")
if not isinstance(args["mask_min_lines"], int) or args["mask_min_lines"] < 0:
    raise Exception("Min. line ammount has an invalid value.")
if not isinstance(args["mask_max_lines"], int) or args["mask_max_lines"] < 0:
    raise Exception("Max. line ammount has an invalid value.")
if not isinstance(args["mask_min_circles"], int) or args["mask_min_circles"] < 0:
    raise Exception("Min. circle ammount has an invalid value.")
if not isinstance(args["mask_max_circles"], int) or args["mask_max_circles"] < 0:
    raise Exception("Max. circle ammount has an invalid value.")

# Load the training images with has the extension .jpg and .png.
# Convert them into RGB and store in an array 
training_images = []
for image_path in os.listdir(args["dataset_path"]):
    if image_path.endswith(".jpg") or image_path.endswith(".png"):
        training_images.append(cv2.cvtColor(cv2.imread(args["dataset_path"] + "/" + image_path, 3), cv2.COLOR_BGR2RGB))

# Check if at least one image to train exists
if len(training_images) == 0:
    raise Exception("The specified training dataset directory is empty.")

# Initalize the GAN (Context Encoder(Generator) and Discriminator) 
network = Network(tiles=args["tiles"], shape=args["shape"])
# Start training
network.train(images=training_images, epochs=args["epochs"], batch_size=args["batch_size"], weights_path=args["weights"], saving_epochs=args["saving_epochs"], mask_min_rectangles=args["mask_min_rectangles"], mask_max_rectangles=args["mask_max_rectangles"], mask_min_lines=args["mask_min_lines"], mask_max_lines=args["mask_max_lines"], mask_min_circles=args["mask_min_circles"], mask_max_circles=args["mask_max_circles"])
