import numpy as np

X = np.array([[0],[5],[0],[12],[8]])
y = np.array([-1, +1, -1, +1, +1])

class DecisionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        preds = np.ones(n_samples)
        feature = X[:, self.feature_idx]

        if self.polarity == 1:
            preds[feature < self.threshold] = -1
        else:
            preds[feature > self.threshold] = -1

        return preds

class AdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, 1 / n_samples)

        for t in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for feature_i in range(n_features):
                feature_values = X[:, feature_i]
                thresholds = np.unique(feature_values)

                for threshold in thresholds:
                    preds = np.ones(n_samples)
                    preds[feature_values < threshold] = -1

                    error = np.sum(w[y != preds])

                    polarity = 1
                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        clf.feature_idx = feature_i
                        clf.threshold = threshold
                        clf.polarity = polarity
                        min_error = error

            if min_error == 0:
                clf.alpha = float('inf')
                self.clfs.append(clf)
                print(f"\nRound {t+1}: Perfect stump found!")
                print("Feature index:", clf.feature_idx)
                print("Threshold:", clf.threshold)
                print("Weighted error ε =", min_error)
                break

            clf.alpha = 0.5 * np.log((1 - min_error) / min_error)

            preds = clf.predict(X)
            w *= np.exp(-clf.alpha * y * preds)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        return np.sign(np.sum(clf_preds, axis=0))

model = AdaBoost(n_clf=5)
model.fit(X, y)

preds = model.predict(X)

print("\nFinal Predictions:", preds)
print("True Labels:     ", y)