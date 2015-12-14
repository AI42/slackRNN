# min-char-rnn in Scala

This is a reimplementation of Andrej Karpathy's min-char-rnn.py code that was written in Python/Numpy.
This reimplementation builds on `Breeze` from Scala NLP and is written in Scala.
Otherwise, it should perform the same.

Original Python code: <https://gist.github.com/karpathy/d4dee566867f8291f086>

# Basic overview

The crux of this model is the `minCharRNN` class, which implements the basic recurrent neural network. The `main()`
method of this class trains the model, while also regularly sampling during training and reporting the progress.