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


# Some infos about hardware and software
Training a Neural Network could be an hard job for your machine.
It has been very difficult with only CPU, but if you invest some time in learning and -you got hardware-, with GPU it is way better and faster. The job is not too much easy: I messed up with the NVIDIA docs, but in the end I've found [this video](https://www.youtube.com/watch?v=hHWkvEcDBO0) enlighting. However you can find some infos on NVIDIA, despite for example my GPU does not appear in the CUDA enabled, but it is.
However some infos:


- Python 3<br>
- PyCharm community edition <br>
- Linux Mint Mmachine<br>
- AMD Ryzen 5 400 series <br>
- NVidia GEFORCE GTX 1650<br>
- 32GB RAM<br>

# Folder structure
You'll find the data you'll need to know:
 - the data are from [this repo](https://github.com/urschrei/lovecraft/blob/master/lovecraft.txt)

main  
  |->scripts<br>
  ||-> models<br>
    ||-> first_lstm.py<br>
    ||-> second_lstm.py<br>
  ||-> chtulhu_bot_3_train.py<br>
  ||-> chtulhu_bot_3_apply.py<br>


The models folder has the specifications of the model tested.
The *train file is the file of training, and the *apply file, the apply file.
Note, you need a weights folder not committed a the same level of scripts, where write the weights of the trained model.

# scripts
Both my scripts are based on some web sources like those:
 - [a raw intro to LSTM](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
 - [how to make better generations](https://stackoverflow.com/questions/47125723/keras-lstm-for-text-generation-keeps-repeating-a-line-or-a-sequence)

You can read well the code on the scripts: the main differences between the two, is that the second has some different setting in epoch and batch and some strata. I've encountered some problem in memory allocation changeing other params.

# some results


Here some result made by chtulhubot2.py. Temperature is a paremater for randomness, the more is high, the most the network generate less probable (and more weird) results. 
The results are kind of gibberish but they seems to capture the sense of the Lovecraft works.

temperature 1.5, input "the creature in the darkness"
alls of the monstrous and the part which the great man was seemed to the few and incalculable margemy strange form of the continuous continued of the things of the strange more and the world of the bearded feet and seemed to the black and contained this decay of the black man of the start of the sight of the warious for a sea of the monnt of the steadier of the signs of the absortage of a strange 

temperature 1.7, input "the creature in the darkness"
former and the other singular and for his deserted stars of the continuous sense of the old college of the care of the singular land of the deserted and dome and toined the deserted and forms of the bottom of the door of the desert of the old narrow and door of the form of all the distance and seemed to condealed a hills and the pictures of the careless deserted soads of a door of male in the door








