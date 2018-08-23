from distutils.core import setup

setup(
    name="KerasSearchCV",
    version="0.1",
    description="Built for the implementation of Keras in Tensorflow. Behaves similarly to GridSearchCV and "
                "RandomizedSearchCV in Sci-Kit learn, but allows for progress to be saved between folds and "
                "for fitting and scoring folds in parallel.",
    py_modules=['SearchCV', 'SearchCVWorker']
)
