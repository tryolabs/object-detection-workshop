# Hands on 2: Luminoth for real world object detection

![Luminoth Logo](https://user-images.githubusercontent.com/270983/31414425-c12314d2-ae15-11e7-8cc9-42d330b03310.png)

## Introduction

In this part of the workshop, we will use the [Luminoth](https://luminoth.ai/) toolkit,
which has an implementation of the Faster R-CNN method we have just seen.

The idea is to learn the usage, and be prepared to solve real world object detection
problems with the tool.

As our case study, we will be building a model able to recognize cars, pedestrians, and
other objects which a self-driving car would need to detect in order to properly function.
We will have our model ready for that and see it how to apply it to images and video. We
will not, however, add any tracking capabilities.

During the workshop session, we will not have time to properly train our model, but we
will nonetheless get to start the training process (albeit, with a toy dataset) and go
over the things you need to look at when training a real model.

### Installation and setting up

To use Luminoth, **TensorFlow** must be installed beforehand.

If you want **GPU support**, which is very desirable given that we are going to run
compute-intensive tasks, you should install the GPU version of TensorFlow first.

We are going to start by creating and activating a **Virtualenv** for the work that
follows:

    python -m venv .virtualenvs/luminoth
    source .virtualenvs/luminoth/bin/activate

Now, we just need to install TensorFlow (GPU version) and Luminoth:

    pip install tensorflow-gpu
    pip install luminoth

To check that everything works as expected, you should run `lumi --help` and get something
like this:

    Usage: lumi [OPTIONS] COMMAND [ARGS]...

    Options:
      -h, --help  Show this message and exit.

    Commands:
      checkpoint  Groups of commands to manage checkpoints
      cloud       Groups of commands to train models in the...
      dataset     Groups of commands to manage datasets
      eval        Evaluate trained (or training) models
      predict     Obtain a model's predictions.
      server      Groups of commands to serve models
      train       Train models

Congratulations! Now Luminoth is setup and you can start playing around.

---

Next: [Using Luminoth 101](/hands-on-2/01-Using-Luminoth-101.md).