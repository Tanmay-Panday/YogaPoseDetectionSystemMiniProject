import subprocess
import streamlit as st
import streamlit.components.v1 as components  # to show pdf
import os
import base64
from PIL import Image
# ACCESS DATA and functions
currDirect = os.path.dirname(os.path.realpath(__file__))
# dataDirect = os.path.join(currDirect, 'PythonCodes and Database')
gbDirect = os.path.join(currDirect, 'GradientBoostingClassifier_metrics.pdf')
gbLearnDirect = os.path.join(currDirect, 'GradientBoostingClassifier_LearningCurve.jpg')
lrDirect = os.path.join(currDirect, 'LogisticRegression_metrics.pdf')
lrLearnDirect = os.path.join(currDirect, 'LogisticRegression_LearningCurve.jpg')
rfDirect = os.path.join(currDirect, 'RandomForestClassifier_metrics.pdf')
rfLearnDirect = os.path.join(currDirect, 'RandomForestClassifier_LearningCurve.jpg')
curvesDirect = os.path.join(currDirect, 'comparison of auc for all roc.jpg')

closeButton = st.button("CLOSE VIEWER")
if closeButton :
    os._exit(0)

def EmptyBufferFunction():
    pass

def display_pdf(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_contents = f.read()
        pdf_base64 = base64.b64encode(pdf_contents).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="700" height="1000" style="border: none;"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
def pressed(pathOP):
    if(st.button("CLOSE PDF")):
        EmptyBufferFunction()
    display_pdf(pathOP)

def display_jpg(jpg_path):
    img = Image.open(jpg_path)
    st.image(img,caption="",use_column_width=True)

def pressed2(pathOP):
    if(st.button("CLOSE JPG")):
        EmptyBufferFunction()
    display_jpg(pathOP)

#FRONT-END

st.title("VIEW THE TRAINING RESULTS")

lrButton = st.button("LOGISTIC RESULTS")
lrLearnButton = st.button("LOGISTIC Learning Curve")
gbButton = st.button("GRADIENT BOOSTING RESULTS")
gbLearnButton = st.button("GRADIENT BOOSTING Learning Curve")
rfButton = st.button("RANDOM FOREST RESULTS")
rfLearnButton = st.button("RANDOM FOREST Learning Curve")
aucButton = st.button("Area Under Curve FOR_All Models")

if(lrButton):
    pressed(lrDirect)

if(lrLearnButton):
    pressed2(lrLearnDirect)

if(gbButton):
    pressed(gbDirect)

if(gbLearnButton):
    pressed2(gbLearnDirect)

if(rfButton):
    pressed(rfDirect)

if(rfLearnButton):
    pressed2(rfLearnDirect)

if(aucButton):
    pressed2(curvesDirect)