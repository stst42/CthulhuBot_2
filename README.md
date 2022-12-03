# CthulhuBot_2
Repository for a generative network for a H.P.Lovecraft text, to study Generative Text Networks.

**Author**<br>
stst42


# Purpose of this repository
Being an useR, despite there're packages in R to manage Keras + Tensorflow, I decided to use them in Python because:

- I wanted to start to learn Python
- I'd like to learn something about Deep Learning

So the first serious approach on Deep Learning has been in Python, with generative networks.
I am a fan of H.P.Lovecraft and I think I've read almost every of his works: so, why not melt my passion for his writings with my passion for machine learning?

So Here we go.

# Purpose of this wall of text
In this write up I'm going to write a kind of diary about what I am doing.

# Some infos about hardware and software
Training a Neural Network could be an hard job for your machine.
It has been very difficult with only CPU, but if you invest some time in learning and -you got hardware-, with GPU it is way better and faster. The job is not too much easy: I messed up with the NVIDIA docs, but in the end I've found [this video](https://www.youtube.com/watch?v=hHWkvEcDBO0) enlighting. However you can find some infos on NVIDIA, despite for example my GPU does not appear in the CUDA enabled, but it is.
However some infos:


- Python 3<br>
- PyCharm community edition <br>
- Windows 11 machine<br>
- AMD Ryzen 5 400 series <br>
- NVidia GEFORCE GTX 1650<br>
- 32GB RAM<br>

# Folder structure
You'll find a couple of folders you'll need to know:
 - the data are from [this repo](https://github.com/urschrei/lovecraft/blob/master/lovecraft.txt)
 - [scripts](https://github.com/stst42/GITGenerativeNetwork/tree/main/scripts) where you can find scripts.

main  
  |->scripts<br>
  ||-> chtulhubot.py<br>
  ||-> chtulhubot.py<br>
  |-> weights-improvement-bigger-35-1.4846.hdf5<br>
  |-> weights-improvement-v2-40-1.4714.hdf5<br>
  |-> weights-improvement-v2-01-2.2795.hdf5<br>

# scripts
Both my scripts are based on some web sources like those:
 - [a raw intro to LSTM](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
 - [how to make better generations](https://stackoverflow.com/questions/47125723/keras-lstm-for-text-generation-keeps-repeating-a-line-or-a-sequence)

You can read well the code on the scripts: the main differences between the two, is that the second has some different setting in epoch and batch and some strata. I've encountered some problem in memory allocation changeing other params.

# weights-improvement-*
Those are already weights of the chtulhubots. The bigger is for the first, the V2 are the first and last epochs weights, with the rispective loss function value.

# some results
Here some result made by chtulhubot2.py.











