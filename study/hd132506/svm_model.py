from sklearn import datasets, svm, metrics
from sklearn.preprocessing import StandardScaler
from loader import load

"""
SVM Classifier which trains 
"""

class SVMModel:
    def __init__(self, C=1.15, gamma=0.001):
        self.classifier = svm.SVC(C=C, gamma=gamma, random_state=108)
        self.trained = False
        
    def train(self, data, label):
        self.classifier.fit(data, label)
        self.trained = True
        
    def predict(self, data):
        assert self.trained, "Modeld hasn't been trained yet."
        return self.classifier.predict(data)


    def evaluate(self, data_size):
        print('---------------------Model Evaluation------------------------')
        test_data, target = load(dataset="testing", data_size=data_size)
        predicted = self.predict(test_data)
        print("Classification report for classifier %s:\n%s\n"
            % (self.classifier, metrics.classification_report(target, predicted)))
        print("Confusion matrix:\n%s" 
            % metrics.confusion_matrix(target, predicted))
        return metrics.classification_report(target, predicted)
