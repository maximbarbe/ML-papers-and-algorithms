import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class LDA():
    def __init__(self):
        super().__init__()
        self.priori_weights = dict()
        self.mu = dict()
        self.std=None
        
    def _multivariate_gaussian(self, x, mu, std):
        return ((2*np.pi)**(std.shape[0]/2) * (np.linalg.det(std))**(1/2))**(-1) * np.exp((-1/2)*np.dot((x-mu).T,np.dot(np.linalg.inv(std),x-mu)))        
        
        
        
    def train(self, x, y):
        classes = np.unique(y)
        N = np.shape(y)[0]
        sigma = np.zeros((x.shape[1], x.shape[1]))
        for c in classes:
            self.priori_weights[c] = np.where(y==c, 1, 0).sum()/N
            class_samples = x[np.where(y==c)[0],:]
            nc = class_samples.shape[0]
            self.mu[c] = class_samples.sum(axis=0)/nc
            for col in class_samples:
                sigma = sigma + np.outer(col-self.mu[c], col-self.mu[c])
        sigma = sigma/x.shape[0]
        self.std = sigma
        
    def predict(self, x):
        if len(self.mu.items()) == 0:
            raise Exception("The model has not been fitted.")
        if x.shape[1] != list(self.mu.values())[0].shape[0]:
            raise Exception("Data isn't the right shape.")
        predictions = []
        classes = list(self.mu.keys())
        for sample in x:
            cur = []
            for c in self.mu.keys():
                cur.append(self.priori_weights[c]*self._multivariate_gaussian(sample, self.mu[c], self.std))
            predictions.append(classes[np.argmax(cur)])
        
        return np.array(predictions)


lda = LDA()

iris = load_iris()
features = iris['data']
target = iris['target']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
lda.train(X_train, y_train)
predictions = lda.predict(X_test)

print((np.where(predictions==y_test, 1, 0).sum()/y_test.shape[0]) * 100)