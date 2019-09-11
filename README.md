# ***This repository is deprecated and does not work as planned. [The new version is released on the Inpainting-using-Context-Encoder-in-Generative-Adversarial-Network-V2 repo.](https://github.com/StevenCyb/Inpainting-using-Context-Encoder-in-Generative-Adversarial-Network-V2)***


# Inpainting using Context-Encoder and Generative-Adversarial-Network
I was excited about the inpainting project from [here](https://github.com/MingtaoGuo/ContextEncoder_Cat-s_head_Inpainting_TensorFlow), which in turn is based on the paper [Context Encoders: Feature Learning by Inpainting](https://arxiv.org/abs/1604.07379). Therefore I wanted to try my own implementation.
Some parts are from the other project, so have a look at the other project and the original paper.
In addition, I also added some own modifications, e.g. an additional layer in the decoder or the tiling of the pictures.
# Table Of Contents
- [Small Evaluation](#small_evaluation)
- [How To Use It](#how_to_use_it)
<a name="small_evaluation"></a>
## Small Evaluation
The following result needed 1:10h with batch size of 1 and 30k epochs on a GTX1080TI.
![Training_vis](/media/training.gif)
![results](/media/results.png)
<a name="how_to_use_it"></a>
## How To Use It
First I created the directory `weights`. Then I copied the training data into the directory `training`.
For the upper training I used the following command:
```
python3 train.py -mar 3 -mal 0 -mac 0 -bs 1
```
The arguments `-mar` define the max. ammount of random ractangles for the random mask.
In addition I disable the random drawing of lines and circles by setting `-mal` and `-mac` to `0`.
But you can play around.

Then I run the following command to perform a prediction:
```
python3 predict.py -i ./media/0.jpg -m ./media/0_mask.jpg -o ./media/0_prediction.jpg
```
The Input images and masks looked like the `Masked-Image` and `Mask` from the upper figure.
Where the white area characterised the region to inpaint.

You can get more information about available arguments by using the argument `-h`.
