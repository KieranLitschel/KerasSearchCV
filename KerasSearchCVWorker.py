import tensorflow as tf
import tensorflow.keras as keras
import os
import sys
import dill
from tensorflow.keras.callbacks import TensorBoard
import datetime

additional_import_file = sys.argv[1]
additional_import = sys.argv[2]
dillPath = sys.argv[3]
thread_number = int(sys.argv[4])

if additional_import_file != '':
    import importlib

    sys.path.append(additional_import_file)

    importlib.import_module(additional_import)

with open(dillPath, 'rb') as handle:
    toDo = dill.load(handle)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = toDo.seed
tf.set_random_seed(seed)

memory_frac = toDo.memory_frac
keras.backend.clear_session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_frac)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

job, fold = toDo.getJob(thread_number)
trainX, trainY, testX, testY = toDo.getTrainTest(fold)
model_constructor = toDo.model_constructor
tensorboard_on = toDo.tensorboard_on

model = keras.wrappers.scikit_learn.KerasClassifier(model_constructor, **job, verbose=0)
if tensorboard_on:
    nice_folder = toDo.tensorboard_folder
    nice_sub_folder = ""
    for key in job.keys():
        val = str(job[key])
        if val[0] == "<" and val[-1] == ">":
            func_comps = val.split(" ")
            nice_val = func_comps[1]
        else:
            nice_val = str(job[key])
        nice_sub_folder += key + "_" + nice_val + "_"
    nice_folder += "/"
    for c in nice_sub_folder:
        if c == ".":
            c = "_POINT_"
        elif c == "-":
            c = "_NEGATIVE_"
        nice_folder += c
    if toDo.cv != 1:
        nice_folder += "/fold_" + str(fold)
    tensorboard = TensorBoard(log_dir='/KerasSearchCV/' + nice_folder + "/", histogram_freq=0,
                              write_graph=True, write_images=True)
    model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=[tensorboard])
else:
    model.fit(trainX, trainY)
acc = model.score(testX, testY)

sys.stdout.write(str(acc))
