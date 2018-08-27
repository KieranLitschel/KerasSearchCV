# Introduction

Whilst tuning hyperparameters in Google Cloud Compute I discovered two problems with using the implementations of RandomSearchCV and GridSearchCV supplied in the Sci-Kit Learn library. So I decided to write my own implementations to overcome these problems.

# Problems with the implementation of searches in Sci-Kit Learn and how my functions overcomes them

## Problem 1: Some neural networks under utilize the GPU

I've found that the neural networks I've been working with only utilize at maximum about 40% of the GPU during training and scoring. This is a problem as using Google Compute you are charged by the second for resources, not by how much of them you utilize them, meaning that under utilizing the GPU is effectively thowing money down the drain. I experimented with fitting multiple models on one GPU, running them in seperate scripts and allocating each an equal amount of memory on the GPU, and discovered that it is quicker to split the models between scripts and train them in parallel than train them all using a single script.

This set of functions effectively automates this process, with the user specifying the number of threads they would like to use and what models they would like to train and score, and then the program creates that many threads, with each thread creating a subprocess which runs the script to train and score a model. The threads share a pickle file which contains an object that manages the models that need training and scoring and which thread is doing what. This is necessary as each of the subprocesses need to be able to read what model they should train and score, and subprocesses can't be passed objects, so this is the simplest solution. This solution is even more effective as each thread is not only assigned a model to train but also a fold, meaning that different folds of the same model are trained in parallel, so cross-validating a specific model is quicker.

## Problem 2: Instances terminating during maintenance events

CPUs can be live migrated in Google Cloud, but GPUs can't, so whenever there is a maintenance event Google Compute must restart the VM. This has meant several times I've lost tens of hours of work, as with the implementation of RandomSearchCV and GridSearchCV in Sci-Kit Learn, no results are given until the search completes.

To overcome this, in my implementation every time a fold is completed the accuracy achieved is written to a pickle file, and this pickle file is also keeping track of which thread is doing what as described previously. This means if a VM is suddenly terminated, the accuracies that had already been computed and the folds of each model that have not are presevered in the pickle file. So all you have to do is setup a startup script to reload the model and it'll continue from where it left off, meaning no human intervention is required and much less progress is lost.

As the set of functions minimize progress lost when a VM terminated, they would also work well for running searches on preemtible devices too, allowing even more money to be saved.

In addition you can manually kill the search by typing "quit" in the command line, and then all the subprocesses and threads are safely killed. The progress is saved in the pickle file as it would be if the process was suddenly terminated, meaning it can be continued where it left off when it is next launched. This is useful for running searches on machines in the background whilst you do other stuff, as it means if you need to use the GPU or turn off the machine you can do so easily terminate the search without losing all of the work already done.

The only problem is that it does not save the progress of models that were been trained when the subprocesses were killed, meaning that you can still lose a lot of work if you are training and scoring a complicated model that takes many hours to train and score each fold. This could be solved by saving progress after a fixed number of epochs, although I have not implemented this as it is much more complicated and I am yet to require it.

# How to install

To install, simply use "python -m pip install git+https://github.com/KieranLitschel/KerasSearchCV.git".

# How to use

## Creating a new search

First create an instance of the class host in KerasSearchCV, the argument "curr_dir" should be the directory where you want data about the search to be stored, in it the program will create a folder called KerasSearchCV, where it will store the pickle file keeping track of the search, all Tensorboard log files (if applicable), and checkpoint saves. The argument "pickle_file" should be the name (with extension) of the file you want the results to be stored in (note that it does not have to exist yet), reload should be set to false here as we are creating a new search.

Next we need to call the instances method create_new, and supply it the follow arguments:

* trainX - Should be an NxM numpy array (where there are N samples and M features) of the training samples without their labels.
* trainY - Should be an NxK numpy array (where N is equal to that in trainX and there are K classes) of the binary sample labels (use keras.utils.to_categorical to convert numerical labels to binary if you have not done so already)
* model_constructor - Should be given a reference to a build function which constructs, compiles and returns a Keras model.
* search_type - Should be "grid" for a grid search, "random" for a randomized search, and "custom" for a custom search
* param_grid
  * If search_type is "grid", then it should be a parameter grid like you would pass GridSearchCV in Scikit-Learn.
  * If search_type is "random", then it should be a parameter distribution like you would pass RandomizedSearchCV in Scikit-Learn.
  * If search_tpye is "custom", then it should be a list of dictionaries, where each dictionary is a set of parameters you'd like to investigate
* cv - The number of folds you want to do of cross-validation for each model, you can train on the whole training set and test on your own validation set if you set this to 1
* threads - The number of models you want to train and score in parallel, it is worth experimenting with this number to see what suits your model and GPU best.
* total_memory - The fraction of memory on the GPU that should be allocated to the search, note if you get an error messages whose traceback starts "CUBLAS_STATUS_ALLOC_FAILED", you have set the fraction too high, and should try a lower number.
* seed - The seed passed to make results repeatable, by default it is 0
* validX & validY - Pass your own validation set if you want to train on the whole testing set (also make sure to set CV to 1), otherwise leave these as None
* tensorboard_on - Set this to true if you would like to enable tensorboard
* epoch_save_period - How often (measured in epochs) the search should create a checkpoint when training each model. Note that if you are also using Tensorboard, if this value is set to greater than 1, when you resume the search the Tensorboard graph will have an overlap as it will most likely have recorded epochs since the save. If you would like to avoid overlaps entirely set the epoch_save_period to 1.

Finally we need to call the instances method start, which starts the search. If at any point you want to quit the search simply type "quit" and press enter. Also note that once the search has finished, you will be informed, and then will need to enter quit to kill all the threads.

## Resume a search

Resuming a search is very similair to creating one. You should create an instance of host, passing it the path of the pickle file that you set at the start of the search you want to resume, and then set reload to True. Then you just need to call start and the search will resume.

## Getting the results

At the end of the search you can get the results be calling getResults on the instance. But if at any point you would like to access the results, simply quit the search, and create a new instance in the same way you would if you were resuming a search, but instead of calling start, call getResults, and the results so far will be returned.

## Explanation of the folder structure managed by the program

The program stores data about searches in the directory you specify when you create the Host object. Note that all references within the program are relative to the KerasSearchCV folder, so if you want to you can move the KerasSearchCV folder, and as long as you specify the new directory you have placed it in when you create the Host object there will not be any issues.

When you create a search for the first time it creates a folder in KerasSearchCV, the name relates to the date and time the search was started, and this folder is used for the whole search (even if you stop and then continue the search). Within this there is a folder for each model being trained, the name of each folder is in the format of \[parameter_name\]\_\[parameter_value\]\_ for each parameter you specified for the search to explore. If cv is set to greater than 1 then within these folders there will be a folder for each fold, if cv is set to 1 then there will be no more subfolders, and all info about the single fold will be stored here. Within the final folder there is the checkpoint save files, labeled "cp-N.cktp" where N is the number of epochs that had been completed when the checkpoint was saved. The checkpoint save contains the topology of the model, the weights at that epoch, and the state of the optimizer, so you can use this checkpoint file to train for further epochs, or just load the weights from it to use for deployment.
