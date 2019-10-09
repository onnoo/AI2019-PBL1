from loader import load
from svm_model import SVMModel

model = SVMModel()

training, targets = load(data_size=1000)

model.train(training, targets)

model.evaluate(10000)