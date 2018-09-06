import sys
import dill
import time

dillPath = sys.argv[3]
thread_number = int(sys.argv[4])

loaded = False
while not loaded:
    try:
        with open(dillPath, 'rb') as handle:
            toDo = dill.load(handle)
        loaded = True
    except dill.UnpicklingError as e:
        loaded = False
        time.sleep(1)
    except EOFError as e:
        loaded = False
        time.sleep(1)

seed = toDo.seed

job, fold = toDo.getJob(thread_number)
trainX, trainY, testX, testY = toDo.getTrainTest(fold)
raw_classifier = toDo.model_constructor

del toDo

model = raw_classifier(**job, random_state=seed)
model.fit(trainX, trainY)
acc = model.score(testX, testY)

sys.stdout.write(str(acc))
