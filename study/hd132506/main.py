from loader import load
from svm_model import SVMModel

model = SVMModel(C=1.75, gamma=0.001)

training, targets = load(data_size=6000)

model.train(training, targets)

model.evaluate(data_size=10000)