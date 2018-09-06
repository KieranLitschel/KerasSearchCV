import inspect
import os
import subprocess
import sys
import threading
import time
from subprocess import PIPE
import datetime

import dill
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler


class ToDo:
    def __init__(self, model_constructor, cv, jobs, trainX, trainY, threads, curr_dir, total_memory=0.8, seed=0,
                 validX=None, validY=None, tensorboard_on=False, epoch_save_period=5, custom_object_scope=None,
                 histogram_freq=0, library="Keras"):
        self.model_constructor = model_constructor
        self.cv = cv
        self.trainX = trainX
        self.trainY = trainY
        self.validX = validX
        self.validY = validY
        self.seed = seed
        self.folds = []
        self.doing = [None] * threads
        self.total_memory = total_memory
        self.threads = threads
        self.memory_frac = total_memory / threads
        self.curr_dir = curr_dir
        self.tensorboard_on = tensorboard_on
        self.tensorboard_folder = ""
        self.epoch_save_period = epoch_save_period
        self.custom_object_scope = custom_object_scope
        self.histogram_freq = histogram_freq
        self.library = library
        self.raw_results = []

        for c in str(datetime.datetime.now()):
            if c == "-" or c == " " or c == "." or c == ":":
                c = "_"
            self.tensorboard_folder += c

        self.job_str_tracker = []
        for job in jobs:
            self.job_str_tracker.append((job, str(job)))

        if inspect.getfile(model_constructor) != "<input>":
            additional_import_file = inspect.getfile(model_constructor)
            paths = additional_import_file.split("\\")
            self.additional_import_dir = "\\".join(paths[0:len(paths) - 1])
            self.additional_import = inspect.getmodulename(additional_import_file)
        else:
            self.additional_import_file = ""
            self.additional_import = ""

        if cv != 1:
            np.random.seed(seed)
            kf = KFold(n_splits=cv, random_state=seed)
            for train, test in kf.split(trainX):
                self.folds.append((train, test))
        else:
            if validX is None or validY is None:
                print("WARNING: Set to do 1 fold CV but validation set is not supplied / fully supplied")

        self.jobs = []
        for job in jobs:
            for i in range(0, cv):
                self.jobs.append((job, i))
        self.jobs.reverse()

        self.accuracies = {}
        for (job, fold) in self.jobs:
            self.accuracies[str(job)] = []
        self.results = {}

    def setNextJob(self, thread_number):
        if len(self.jobs) > 0:
            job, fold = self.jobs.pop()
            self.doing[thread_number] = (job, fold)
        else:
            self.doing[thread_number] = None
            done = True
            for job in self.doing:
                if job is not None:
                    done = False
                    break
            if done:
                print("-Finishing up last fold, once fold completed message is recieved, type \"quit\" to exit")
        return self.doing[thread_number]

    def getJob(self, thread_number):
        return self.doing[thread_number]

    def doneJob(self, thread_number, accuracy):
        job, _ = self.getJob(thread_number)
        self.accuracies[str(job)].append(accuracy)
        if len(self.accuracies[str(job)]) == self.cv:
            result = []
            for key in job.keys():
                result.append(str(job[key]))
            if self.cv != 1:
                accs = np.array(self.accuracies[str(job)])
                mean = np.average(accs)
                std = np.std(accs)
                for acc in accs:
                    result.append(str(acc))
                result.append(str(std))
                result.append(str(mean))
                self.results[str(job)] = {'mean': mean, 'std': std, 'accs': accs}
                print(
                    "---Got mean of " + ("%.6f" % mean) + " and std of " + ("%.6f" % std) + " with parameters " + str(
                        job))
            else:
                self.results[str(job)] = {'acc': self.accuracies[str(job)][0]}
                result.append(str(self.accuracies[str(job)][0]))
                print(("---Got accuracy of %.6f" % self.accuracies[str(job)][0]) + " with parameters " + str(job))
            self.raw_results.append(result)
        self.doing[thread_number] = None

    def prepare_for_reload(self):
        for job_fold in self.doing:
            if job_fold is not None:
                self.jobs.append(job_fold)
        for i in range(0, len(self.doing)):
            self.doing[i] = None
        for i in range(0, len(self.job_str_tracker)):
            job, old_job_str = self.job_str_tracker[i]
            if (str(job)) != old_job_str:
                if self.accuracies.get(old_job_str) is not None:
                    self.accuracies[str(job)] = self.accuracies[old_job_str]
                    del self.accuracies[old_job_str]
                if self.results.get(old_job_str) is not None:
                    self.results[str(job)] = self.results[old_job_str]
                    del self.results[old_job_str]
                self.job_str_tracker[i] = (job, str(job))

    def setNumberOfThreads(self, threads=None, total_memory=None):
        self.prepare_for_reload()
        if total_memory is not None:
            self.total_memory = total_memory
        if threads is not None:
            self.threads = threads
        self.doing = [None] * self.threads
        self.threads = self.threads
        self.memory_frac = self.total_memory / threads

    def getTrainTest(self, fold):
        if self.cv != 1:
            trainX = self.trainX[self.folds[fold][0]]
            trainY = self.trainY[self.folds[fold][0]]
            testX = self.trainX[self.folds[fold][1]]
            testY = self.trainY[self.folds[fold][1]]
        else:
            trainX = self.trainX
            trainY = self.trainY
            testX = self.validX
            testY = self.validY
        return trainX, trainY, testX, testY

    def getFullPath(self, relativePath):
        return self.curr_dir + "\\" + relativePath


class WorkerThread(threading.Thread):
    def __init__(self, thread_number, dillPath):
        threading.Thread.__init__(self)
        self.thread_number = thread_number
        self.dillPath = dillPath
        self.pythonPath = sys.executable

    def run(self):
        global kill_flag
        writePickleLock.acquire()
        with open(self.dillPath, 'rb') as handle:
            toDo = dill.load(handle)
        additional_import_dir = toDo.additional_import_dir
        additional_import = toDo.additional_import
        nextJob = toDo.setNextJob(self.thread_number)
        if toDo.library == "Keras":
            worker_path = "KerasSearchCVWorker.py"
        elif toDo.library == "SKLearn":
            worker_path = "SKLearnSearchCVWorker.py"
        with open(self.dillPath, 'wb') as handle:
            dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
        del toDo
        writePickleLock.release()
        if nextJob is None:
            more_jobs = False
        else:
            more_jobs = True
        while more_jobs and kill_flag is False:
            changeProcsLock.acquire()
            if kill_flag:
                changeProcsLock.release()
                break
            print("-Starting fold " + str(nextJob[1] + 1) + " of job " + str(nextJob[0]))
            start = time.time()
            proc = subprocess.Popen(
                [self.pythonPath, os.path.join(dir_path, worker_path), additional_import_dir,
                 additional_import,
                 self.dillPath, str(self.thread_number)], stdout=PIPE, stderr=PIPE)
            procs[self.thread_number] = proc
            changeProcsLock.release()
            output, err = proc.communicate()
            output = output.decode("utf-8")
            err = err.decode("utf-8")
            if kill_flag:
                break
            if proc.returncode != 0:
                print("Error encountered whilst scoring and fitting a model, please enter quit and reload the search")
                print(err)
                kill_flag = True
                break
            acc = float(output)
            writePickleLock.acquire()
            with open(self.dillPath, 'rb') as handle:
                toDo = dill.load(handle)
            toDo.doneJob(self.thread_number, acc)
            oldJob = nextJob
            nextJob = toDo.setNextJob(self.thread_number)
            if nextJob is None:
                more_jobs = False
            with open(self.dillPath, 'wb') as handle:
                dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
            del toDo
            print("--Finished fold " + str(oldJob[1] + 1) + " of job " + str(oldJob[0]) + (
                    " it took %.1f minutes" % ((time.time() - start) / 60)))
            writePickleLock.release()


class Host:
    def __init__(self, curr_dir="", pickle_file="KSCV.dill", reload=False):
        global dir_path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        global writePickleLock
        writePickleLock = threading.Lock()
        global changeProcsLock
        changeProcsLock = threading.Lock()
        self.curr_dir = curr_dir + "\\KerasSearchCV\\"
        if not os.path.isdir(self.curr_dir):
            os.makedirs(self.curr_dir)
        self.dillPath = pickle_file
        self.full_dill_path = self.curr_dir + self.dillPath
        self.file_found = False
        if reload:
            try:
                with open(self.full_dill_path, 'rb') as handle:
                    toDo = dill.load(handle)
                toDo.prepare_for_reload()
                toDo.curr_path = self.curr_dir
                self.thread_count = toDo.threads
                with open(self.full_dill_path, 'wb') as handle:
                    dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
                self.file_found = True
            except FileNotFoundError:
                print("Error: Could not find the file at " + self.full_dill_path)

    def create_new(self, trainX, trainY, model_constructor, search_type, param_grid,
                   iterations=10, cv=4, threads=2, total_memory=0.8, seed=0, validX=None, validY=None,
                   tensorboard_on=False, epoch_save_period=5, custom_object_scope=None, histogram_freq=0):
        create = False
        try:
            with open(self.full_dill_path, 'rb') as handle:
                toDo = dill.load(handle)
        except FileNotFoundError:
            create = True
        if create is False:
            msg = ""
            while msg != "y" and msg != "n":
                print(
                    "A file with that name already exists, would you like to overwrite it? Answer y for yes and n for no.")
                msg = input()
                if msg == 'y':
                    print("Overwriting the old file.")
                    create = True
                    break
                elif msg == 'n':
                    print(
                        "The file will not be overwritten, please create a new object or change the path attribute in "
                        "this object to point to a different file before calling this method again.")
                    break
        if create:
            if search_type == 'custom':
                jobs = param_grid
            elif search_type == 'grid':
                jobs = list(ParameterGrid(param_grid))
            elif search_type == 'random':
                jobs = list(ParameterSampler(param_grid, iterations, seed))
            toDo = ToDo(model_constructor, cv, jobs, trainX, trainY, threads, self.curr_dir, total_memory, seed, validX,
                        validY, tensorboard_on, epoch_save_period, custom_object_scope, histogram_freq, library="Keras")
            with open(self.full_dill_path, 'wb') as handle:
                dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
            self.thread_count = threads
            self.file_found = True

    def create_new_sklearn(self, trainX, trainY, raw_classifier, search_type, param_grid, iterations=10, cv=4,
                           threads=2, seed=0, validX=None, validY=None):
        create = False
        try:
            with open(self.full_dill_path, 'rb') as handle:
                toDo = dill.load(handle)
        except FileNotFoundError:
            create = True
        if create is False:
            msg = ""
            while msg != "y" and msg != "n":
                print(
                    "A file with that name already exists, would you like to overwrite it? Answer y for yes and n for no.")
                msg = input()
                if msg == 'y':
                    print("Overwriting the old file.")
                    create = True
                    break
                elif msg == 'n':
                    print(
                        "The file will not be overwritten, please create a new object or change the path attribute in "
                        "this object to point to a different file before calling this method again.")
                    break
        if create:
            if search_type == 'custom':
                jobs = param_grid
            elif search_type == 'grid':
                jobs = list(ParameterGrid(param_grid))
            elif search_type == 'random':
                jobs = list(ParameterSampler(param_grid, iterations, seed))
            toDo = ToDo(raw_classifier, cv, jobs, trainX, trainY, threads, self.curr_dir, seed=seed, validX=validX,
                        validY=validY, library="SKLearn")
            with open(self.full_dill_path, 'wb') as handle:
                dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
            self.thread_count = threads
            self.file_found = True

    def change_threads_memory(self, threads=None, total_memory=None):
        if self.file_found is True:
            if total_memory is not None or threads is not None:
                with open(self.full_dill_path, 'rb') as handle:
                    toDo = dill.load(handle)
                toDo.setNumberOfThreads(threads, total_memory)
                with open(self.full_dill_path, 'wb') as handle:
                    dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
            if threads is not None:
                self.thread_count = threads
        else:
            print("Error: You must create or load a search before doing this")

    def start(self):
        if self.file_found is True:
            global procs
            procs = [None] * self.thread_count
            threads = []
            global kill_flag
            kill_flag = False
            try:
                for thread_no in range(0, self.thread_count):
                    thread = WorkerThread(thread_no, self.full_dill_path)
                    thread.start()
                    threads.append(thread)
                msg = ""
                while msg != "quit":
                    print("Type quit at any time to end the search.")
                    msg = input()
            finally:
                print("Making sure all processes are killed, please be patient")
                changeProcsLock.acquire()
                kill_flag = True
                for proc in procs:
                    if proc is not None:
                        proc.kill()
                changeProcsLock.release()
                for thread in threads:
                    thread.join()
                print("All processes safely killed")
        else:
            print("Error: You must create or load a search before doing this")

    def setCustomObjectScope(self, custom_object_scope):
        if self.file_found is True:
            with open(self.full_dill_path, 'rb') as handle:
                toDo = dill.load(handle)
            toDo.custom_object_scope = custom_object_scope
            with open(self.full_dill_path, 'wb') as handle:
                dill.dump(toDo, handle, protocol=dill.HIGHEST_PROTOCOL, byref=False, recurse=True)
        else:
            print("Error: You must create or load a search before doing this")

    def getResults(self):
        if self.file_found is True:
            with open(self.full_dill_path, 'rb') as handle:
                toDo = dill.load(handle)
            return toDo.results
        else:
            print("Error: You must create or load a search before doing this")

    def resultsToCSV(self):
        if self.file_found is True:
            with open(self.full_dill_path, 'rb') as handle:
                toDo = dill.load(handle)
            lines = ""
            for result in toDo.raw_results:
                line = ""
                for element in result:
                    line += element + ","
                line = line[0:len(line) - 1]
                lines += line + "\n"
            dateStr = ""
            for c in str(datetime.datetime.now()):
                if c == "-" or c == " " or c == "." or c == ":":
                    c = "_"
                dateStr += c
            with open(toDo.curr_dir + 'SearchResults_%s.csv' % dateStr, 'w') as f:
                f.write(lines)
                f.close()
            print(
                "Created CSV file with results in current directory with file name %s" % 'SearchResults_%s.csv' % dateStr)
        else:
            print("Error: You must create or load a search before doing this")
