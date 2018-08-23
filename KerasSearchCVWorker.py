import tensorflow as tf
import tensorflow.keras as keras
import os
import sys
import dill

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

model = keras.wrappers.scikit_learn.KerasClassifier(model_constructor, **job, verbose=0)
model.fit(trainX, trainY)
acc = model.score(testX, testY)

sys.stdout.write(str(acc))
