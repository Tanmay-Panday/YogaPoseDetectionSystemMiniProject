import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("D:\Yoga Pose with csvData\webApp\PythonCodes and Database\Database\coordinates.csv")

x=df.drop(['class'], axis=1)   # features
y=df['class']   # target values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1234)

def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, clf):
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  # Set zero_division to 1
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Metrics for {clf.__class__.__name__}:")
    print(f"  Accuracy: {accuracy}")
    print(f"  Precision: {precision}")
    print(f"  Recall: {recall}")
    print(f"  F1 Score: {f1}")

    return model, accuracy, clf.__class__.__name__

def plot_confusion_matrix(model, X_test, y_test, class_labels, clf_name):
    y_pred = model.predict(X_test)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                        display_labels=class_labels)
    cm_display.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    plt.title(f'Confusion Matrix - {clf_name}')
    plt.show()

def save_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"The model has been saved as {filename}")

classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}
best_model = None
best_accuracy = 0.0
best_classifier_name = None

for clf_name, clf in classifiers.items():
    model, accuracy, classifier_name = train_and_evaluate_classifier(X_train, X_test, y_train, y_test, clf)
    plot_confusion_matrix(model, X_test, y_test, class_labels=df['class'].unique(), clf_name=classifier_name)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_classifier_name = classifier_name

print(f"\nThe best model ({best_classifier_name}) has been selected with accuracy {best_accuracy}")
print(best_model)
save_model(best_model, 'best_model.pkl')