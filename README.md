# min-char-rnn.scala

This is a reimplementation of Andrej Karpathy's `min-char-rnn.py` code that was written in Python/Numpy.
This implementation builds on `Breeze` from Scala NLP <http://www.scalanlp.org/> and is written in Scala.
The goal was to rewrite the model such that it performs the same.

The original Python code can be found here: <https://gist.github.com/karpathy/d4dee566867f8291f086>

## Basic overview

The crux of this model is the `minCharRNN` class, which implements the basic recurrent neural network for learning character distributions. 

The `main()` method of this class trains the model, while also regularly sampling during training and reporting progress.
The `lossFunction()` method implements the forward and backward passes in training, while `sample()` performs a forward
pass from a given "seed" character.

## Dependencies

The network is written in Scala `2.11.7` and it builds on the `Breeze` package from Scala NLP. As such, it requires a 
JVM (`1.7` was used in developing this version). The provided `build.sbt` file should handle the `Breeze` dependency.

## Parsing .JSON files from Slack

This network has been developed primarily to learn conversations from Slack. For this, we need to parse Slack archives.
These are stored in JSON files. The `slackParser` class implements this parser. The class instantiation takes three `String` arguments:
where the json archives are (a folder), the name of the file with users listed, and the name of the file with channels listed.
All these are standard Slack archive JSON files you can obtain by requesting an archive of conversations of your Slack team.
The channel and user JSON files should be kept separate from the archive folder.

The parser creates `.txt` files for each user it finds in the archives, with all the messages this user has ever sent in the file,
stripped of any metadata. 

## Running

Currently, this is still a very barebones implementation. Probably the easiest way to try it out yourself is to download
this repository and build it in IntelliJ IDEA. I will provide a more straightforward way of running it.