from setuptools import setup

setup(
    name="KerasSearchCV",
    version="0.1",
    description="Built for the implementation of Keras in Tensorflow. Behaves similarly to GridSearchCV and "
                "RandomizedSearchCV in Sci-Kit learn, but allows for progress to be saved between folds and "
                "for fitting and scoring folds in parallel.",
    py_modules=['KerasSearchCV', 'KerasSearchCVWorker'],
    install_requires=['absl-py==0.4.0',
                      'astor==0.7.1',
                      'gast==0.2.0',
                      'grpcio==1.14.1',
                      'Markdown==2.6.11',
                      'numpy==1.14.5',
                      'protobuf==3.6.1',
                      'scikit-learn==0.19.2',
                      'six==1.11.0',
                      'tensorboard==1.10.0',
                      'tensorflow-gpu==1.10.0',
                      'termcolor==1.1.0',
                      'Werkzeug==0.14.1',
                      'dill==0.2.8.2']
)
