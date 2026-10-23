A training pipeline for an image watermarking set of neural nets. In particular, two NNs are trained: one encoder and one decoder. The encoder takes in a message and an image and encodes a message into an image by slightly tweaking pixels in places that are least noticeable. Decoder is the reverse - given an image, it tries to see if there is a message in there and if there is one, decodes it.

The default dataset is just a dataset of streetfood from Kaggle but there is nothing stopping one from choosing any other image dataset.
