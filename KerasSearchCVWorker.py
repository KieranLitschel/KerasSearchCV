import tensorflow as tf
import tensorflow.keras as keras
import os
import sys
import dill
from tensorflow.keras.callbacks import TensorBoard
import pathlib
from sklearn.metrics import accuracy_score
from numpy import argmax

additional_import_file = sys.argv[1]
additional_import = sys.argv[2]
dillPath = sys.argv[3]
thread_number = int(sys.argv[4])

if additional_import_file != '':
    import importlib

    sys.path.append(additional_import_file)

    importlib.import_module(additional_import)

loaded = False
while not loaded:
    try:
        with open(dillPath, 'rb') as handle:
            toDo = dill.load(handle)
        loaded = True
    except dill.UnpicklingError as e:
        loaded = False
    except EOFError as e:
        loaded = False

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

nice_folder = toDo.getFullPath(toDo.tensorboard_folder)
nice_sub_folder = ""
for key in job.keys():
    val = str(job[key])
    if val[0] == "<" and val[-1] == ">":
        func_comps = val.split(" ")
        nice_val = func_comps[1]
    else:
        nice_val = str(job[key])
    nice_sub_folder += key + "_" + nice_val + "_"
nice_folder += "\\"
for c in nice_sub_folder:
    if c == ".":
        c = "_POINT_"
    elif c == "-":
        c = "_NEGATIVE_"
    nice_folder += c
if toDo.cv != 1:
    nice_folder += "\\fold_" + str(fold)
nice_folder += "\\"

custom_object_scope = toDo.custom_object_scope
epoch_save_period = toDo.epoch_save_period
histogram_freq = toDo.historgram_freq

del toDo

checkpoints = pathlib.Path(nice_folder).glob("*.ckpt")
checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
if len(checkpoints) == 0:
    initial_epoch = 0
    model = keras.wrappers.scikit_learn.KerasClassifier(model_constructor, **job, verbose=0)
else:
    last_checkpoint = str(checkpoints[-1]) + ".ckpt"
    if custom_object_scope is None:
        model = keras.models.load_model(last_checkpoint)
    else:
        with custom_object_scope:
            model = keras.models.load_model(last_checkpoint)
    initial_epoch = int(str(os.path.basename(last_checkpoint)).split("-")[1].split(".")[0])

checkpoint_path = nice_folder + "cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=False,
                                                 verbose=0, period=epoch_save_period)

# When the model is loaded it only retains weights and topology, meaning we have to treat the model slightly different
if tensorboard_on:
    tensorboard = TensorBoard(log_dir=nice_folder, histogram_freq=histogram_freq,
                              write_graph=True, write_images=True)
    if initial_epoch == 0:
        model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=[tensorboard, cp_callback])
    else:
        model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=job['batch_size'], epochs=job['epochs'],
                  initial_epoch=initial_epoch, callbacks=[tensorboard, cp_callback], verbose=0)
else:
    if initial_epoch == 0:
        model.fit(trainX, trainY, callbacks=[cp_callback])
    else:
        model.fit(trainX, trainY, batch_size=job['batch_size'], epochs=job['epochs'], initial_epoch=initial_epoch,
                  callbacks=[cp_callback], verbose=0)

if initial_epoch == 0:
    acc = model.score(testX, testY)
else:
    predTestY = model.predict(testX)
    acc = accuracy_score(argmax(testY, axis=1), argmax(predTestY, axis=1))

sys.stdout.write(str(acc))
