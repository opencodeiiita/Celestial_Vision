import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import warnings
warnings.filterwarnings('ignore')
class BaselineClassifier:
    def __init__(self, embedding_dim=2048):
        self.embedding_dim = embedding_dim
        self.model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
        self.scaler = StandardScaler()
        self.feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    def extract_features(self, images):
        features = []
        for img in images:
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feat = self.feature_extractor.predict(img, verbose=0)
            features.append(feat.flatten())
        return np.array(features)
    def create_synthetic_data(self, n_samples=500, n_classes=5):
        np.random.seed(42)
        X = np.random.randn(n_samples, self.embedding_dim)
        for i in range(n_classes):
            X[i*100:(i+1)*100] += np.random.randn(1, self.embedding_dim) * 2 + i
        y = np.repeat(np.arange(n_classes), n_samples // n_classes)
        return X[:n_samples], y[:n_samples]
    def train(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train, verbose=0)
    def predict(self, X_test):
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
if __name__ == '__main__':
    classifier = BaselineClassifier()
    X, y = classifier.create_synthetic_data(n_samples=500, n_classes=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)
    print(f'Accuracy: {metrics["accuracy"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1-Score: {metrics["f1"]:.4f}')
