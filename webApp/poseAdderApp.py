import streamlit as st 
import subprocess
import os
import importlib.util
import time

currDirect = os.path.dirname(os.path.realpath(__file__))
adderDirect = os.path.join(currDirect, 'PythonCodes and Database', 'addInDatabase Script', 'add.py')
sorterDirect = os.path.join(currDirect, 'PythonCodes and Database', 'sortDatabase script', 'CSVsorter.py')
trainDirect = os.path.join(currDirect, 'PythonCodes and Database', 'trainAndEvaluateAndSave Script', 'trainAndSave.py')

st.title("Add yout yoga pose")
closeButton = st.button("CLOSE EDITOR")
if closeButton :
    os._exit(0)

backend_Sidebar = st.sidebar
with backend_Sidebar:
    st.title("Backend Output")

def addToBackendOutput(stringOutput, n):
    with backend_Sidebar:
        st.markdown(f'<div style="word-wrap: break-word;">Output {n} :\n{stringOutput}</div>', unsafe_allow_html=True)


nameofPose = st.text_input("PLEASE ENTER THE YOGA POSE")
nameofPose = nameofPose.strip()
if (nameofPose != "" ): 
    st.write("Yoga Pose Entered : ",nameofPose)
    startButton = st.button("Start recording the yoga pose")
    if startButton :
        st.write(os.path.exists(adderDirect))

        result = subprocess.run(['python',adderDirect],input=nameofPose,capture_output=True,text=True)
        op1 = result.stdout
        st.write("YOGA POSE HAS BEEN ADDED TO CSV")
        addToBackendOutput(op1,1)

        result = subprocess.run(['python',sorterDirect],capture_output=True,text=True)
        op2 = result.stdout
        st.write("CSV DIRECTORY IS SORTED NOW")
        addToBackendOutput(op2,2)

        st.write("NOW MODELS ARE BEING TRAINED SO PLEASE BEAR FOR A WHILE...")
        
        result = subprocess.run(['python',trainDirect],capture_output=True,text=True)
        op3 = result.stdout
        st.write("Best Performing Model Have been Successfully Saved")
        addToBackendOutput(op3,3)
else :
    st.write("Yoga pose container is Empty Right now")