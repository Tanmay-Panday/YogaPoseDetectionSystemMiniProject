import pandas as pd
from sklearn.model_selection import train_test_split,learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image
import os
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

script_directory = os.path.dirname(os.path.realpath(__file__))
PythonCodesAndDatabase = os.path.dirname(script_directory)
coordinates_path = os.path.join(PythonCodesAndDatabase, 'Database', 'coordinates.csv')

df = pd.read_csv(coordinates_path)

x = df.drop(['class'], axis=1)   # features
y = df['class']   # target values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

df_train = pd.concat([pd.DataFrame({'class': y_train}), X_train], axis=1)
df_test = pd.concat([pd.DataFrame({'class': y_test}),X_test], axis=1)
cv = 2  # Number of cross-validation folds

def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return scores.mean()

def plot_learning_curve(model, X, y, classifier_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
    )

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.title('Learning Curve')
    plt.legend()

    save_path = f"{classifier_name}_LearningCurve.jpg"
    plt.savefig(save_path)
    print(f"Learning curve saved to {save_path}")

df_train = pd.concat([pd.DataFrame({'class': y_train}), X_train], axis=1)
df_test = pd.concat([pd.DataFrame({'class': y_test}), X_test], axis=1)

train_path = os.path.join(PythonCodesAndDatabase, 'Database', 'train.csv')
test_path = os.path.join(PythonCodesAndDatabase, 'Database', 'test.csv')
df_train.to_csv(train_path, index=False)
df_test.to_csv(test_path, index=False)

def plot_confusion_matrix_buffer(model, X_test, y_test, class_labels):
    y_pred = model.predict(X_test)

    # Extract the classifier name from the pipeline
    classifier_name = model.steps[-1][1].__class__.__name__

    confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                                      display_labels=class_labels)

    # Create a BytesIO buffer to save the confusion matrix plot
    buffer = BytesIO()
    fig, ax = plt.subplots()
    # Plot the confusion matrix to the buffer
    confusion_matrix_display.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)

    # Clear the plot for the next iteration
    plt.clf()

    return buffer


def save_metrics_and_plot_to_pdf(classifier_name, metrics, confusion_matrix_plot):
    # Create a BytesIO buffer to save the PDF content
    buffer = BytesIO()

    # Create a canvas and set the page size
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)

    # Set font and font size
    pdf_canvas.setFont("Helvetica", 12)

    # Write metrics information to the PDF
    pdf_canvas.drawString(50, 750, f"Metrics for {classifier_name}:")
    pdf_canvas.drawString(50, 730, f"  Accuracy: {metrics['Accuracy']} ,   Precision: {metrics['Precision']} ,   Recall: {metrics['Recall']} ,   F1 Score: {metrics['F1 Score']}")
    # pdf_canvas.drawString(50, 710, f"")
    # pdf_canvas.drawString(50, 690, f"")
    # pdf_canvas.drawString(50, 670, f"")

    # Convert BytesIO buffer to PIL image
    confusion_matrix_image = Image.open(confusion_matrix_plot)

    # Adjust the positioning and dimensions
    image_width, image_height = confusion_matrix_image.size
    x_position = 50
    y_position = 370
    target_width = 400  # Adjust as needed
    target_height = (target_width / image_width) * image_height

    # Draw the confusion matrix plot on the PDF
    pdf_canvas.drawInlineImage(confusion_matrix_image, x_position, y_position, width=target_width, height=target_height)

    # Save the PDF content to the buffer
    pdf_canvas.save()

    # Move the buffer cursor to the beginning
    buffer.seek(0)

    # Save the buffer content to a PDF file
    pdf_filename = f"{classifier_name}_metrics.pdf"
    with open(pdf_filename, "wb") as pdf_file:
        pdf_file.write(buffer.read())

    print(f"The metrics and confusion matrix have been saved as {pdf_filename}")


def train_and_evaluate_classifier(X_train, X_test, y_train, y_test, clf):
    model = make_pipeline(StandardScaler(), clf)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model using cross-validation
    cv_accuracy = evaluate_model(model, x, y)
    print(f'Cross-Validation Accuracy: {cv_accuracy}')

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)  # Set zero_division to 1
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    classifier_name = clf.__class__.__name__

    # Plot and save the learning curve
    plot_learning_curve(model, x, y, classifier_name)

    return model, accuracy, clf.__class__.__name__

def save_model(model, filename):
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"\nThe model has been saved as {filename}\n")

def plot_roc_curve_for_all_models(final_models, X_test, y_test, class_labels):
    plt.figure(figsize=(8, 6))

    for model, clf_name in final_models:
        lb = label_binarize(y_test, classes=class_labels) # one hot encoded binary matrix
        n_classes = lb.shape[1] # no. of classes (rows)

        #using OneVsRestClassifier for multi class classifications
        classifier = OneVsRestClassifier(model)

        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        # Compute micro-average ROC curve and ROC area
        fpr, tpr, _ = roc_curve(lb.ravel(), y_score.ravel()) #no need of threshold
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the model
        plt.plot(fpr, tpr, label=f'{clf_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc='lower right')
    plt.savefig("comparison of auc for all roc.jpg")

def trainAndSavePdfsAndModel():
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000,C=0.01), # hyperparameter tuning  regularization (after gird search)
        'Random Forest': RandomForestClassifier(n_estimators=50), # hyperparameter tuning reducing decision trees (after gird search)
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50) # hyperparameter tuning reducing decision trees (after gird search)
    }
    best_model = None
    best_accuracy = 0.0
    best_classifier_name = None
    final_Models = [] # to store trained models with their classifier name in a pair

    for clf_name, clf in classifiers.items():
        model, accuracy, classifier_name = train_and_evaluate_classifier(X_train, X_test, y_train, y_test, clf)
        final_Models.append([model,classifier_name])
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_classifier_name = classifier_name

        # Revert to using original class labels
        save_metrics_and_plot_to_pdf(classifier_name, {
            'Accuracy': accuracy,
            'Precision': precision_score(y_test, model.predict(X_test), average='weighted', zero_division=1),
            'Recall': recall_score(y_test, model.predict(X_test), average='weighted'),
            'F1 Score': f1_score(y_test, model.predict(X_test), average='weighted')
        }, plot_confusion_matrix_buffer(model, X_test, y_test, class_labels=df['class'].unique()))

    print(f"\nThe best model ({best_classifier_name}) has been selected with accuracy {best_accuracy}\n")
    print(best_model)
    save_model(best_model, 'best_model.pkl')
    plot_roc_curve_for_all_models(final_Models, X_test, y_test, class_labels=df['class'].unique())
print("\ntraining started")
trainAndSavePdfsAndModel()
print("\ntraining ended")