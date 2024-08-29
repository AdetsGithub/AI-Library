# AI-Library

<p align="center">
  <img src="https://udayton.edu/magazine/2021/01/images/2101_neuralnetwork_card.jpg" />
</p>

This is a project to make a simple, encapsulated and well commented and documented framework for making neural nets. As far as possible, I adhere to OOP paradigms and intend the library as a soft introduction to neural networks.

## Getting started

You will need to install some dependencies for the code to run. These are minimal as code is mostly just vanilla python, NumpPy and graphing libraries. These can be installed by running

```
pip install -r requirements.txt
```

A write up of how the library is structured and some relevant mathematics can be found [here](https://github.com/AdetsGithub/AI-Library/blob/main/AI_Library.pdf)

Some examples solved using the library can be found [here](https://github.com/AdetsGithub/AI-Library/tree/main/Examples)

## Documentation

Documentation is auto-generated using PyDoc. Navigate to AI-Library directory and run
```
python3 -m pydoc -p 3000
```
or
```
python -m pydoc -p 3000
```
depending on operating system.

## TODO

- [x] Documentation (as of 12/07/24)
- [x] Writeup (as of 14/07/24)
- [ ] Convnet implementation
- [ ] RNN implementation
