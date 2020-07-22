This is an attempt to picture what the various parts of go-playing neural networks do.

One method is to take a neuron (or a whole channel) and find inputs that cause a high (or low) activation of the neuron. In the case of image-classifying networks, the derivative of the output with respect to the inputs helps find a good example image. Networks used for go are typically convolutional as well, but the input is discrete. Gradient ascent can't be used unless we accept arbitrary fractional inputs, which might be hard to interpret.

`StochasticBoard` tries to overcome this with probability distributions, which are continuous even if the sample space is discrete. Inputs are optimized by taking random samples instead of differentiating the network.

`model.py` and `board.py` are taken from [katago](https://github.com/lightvector/KataGo/).

This project uses TensorFlow 1.
Run with `python3 govis.py`.