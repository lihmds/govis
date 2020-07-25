This is an attempt to picture what the various parts of go-playing neural networks do.

One method is to take a neuron (or a whole channel) and find inputs that cause a high (or low) activation of the neuron. In the case of image-classifying networks, the derivative of the output with respect to the inputs helps find a good example image. Networks used for go are typically convolutional as well, but the input is discrete. Gradient ascent can't be used unless we accept arbitrary fractional inputs, which may be hard to interpret.

`StochasticBoard` tries to overcome this with probability distributions, which are continuous even if the sample space is discrete. Input optimization relies on random samples instead of the network's derivative. Random sample boards are used to visualize the progress of the optimization as well.

At present the program focuses on [KataGo](https://github.com/lightvector/KataGo)'s networks. `model.py` and `board.py` are taken verbatim from its source code.

# Setup

1. Clone this repository with `git clone https://github.com/lihmds/govis.git`.
2. [Install pipenv](https://pipenv.pypa.io/en/latest/install/#installing-pipenv) if necessary. One way is `pip3 install --user pipenv`.
3. Take care of the dependencies with `pipenv install --three`.

# Running

Run with `pipenv run python govis.py`.

By default, a corner neuron of a small network is visualized. Soon you should see that black stones (marked as `X`) dominate the upper left corner. Feel free to experiment with `parameters.py`, especially `network_path`, `neuron_location` and `hyperparameters`.