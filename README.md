<!---Creating a Model--->
1. Run train.py first, and input the necessary information.
2. Seed refers to the RNG of the program.
3. Epochs are the runs of the training program. More Epochs mean better accuracy, but takes much longer (my computer ran 1 epoch per hour)
4. Number of Batches refer to the processing speed. A higher batch size means the program will run each epoch faster. However it will require much more out of your computer. Unless you run tensorflow using a video card, I do not recommend a batch size larger than 16.
5. Once training is finished, a files folder with a
