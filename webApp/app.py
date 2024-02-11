import streamlit as st 
import subprocess
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

# All paths
currDirect = os.path.dirname(os.path.realpath(__file__))  ## WEBAPP
creatorDirect = os.path.join(currDirect, 'PythonCodes and Database', 'checkAndCreateDatabase Script', 'checkAndCreate.py')
databaseDirect = os.path.join(currDirect, 'PythonCodes and Database', 'Database', 'coordinates.csv')
adderStreamlitDirect = os.path.join(currDirect, 'poseAdderApp.py')
resultsStreamlitDirect = os.path.join(currDirect, 'viewPdfsApp.py')
dataframesStreamlitDirect = os.path.join(currDirect, 'viewDataApp.py')

predictorDirect = os.path.join(currDirect, 'PythonCodes and Database', 'predictPose Script', 'predictor.py')

def runScript_And_return_TextOutput(path):
    result = subprocess.run(['python',path], capture_output=True,text=True)
    return result.stdout

def runAnotherStreamlitApp(path):
    subprocess.run(['streamlit','run',path])

def EmptyBufferFunction():
    pass


def main():
    st.title("Yoga Pose Detection System")
    st.sidebar.title("Navigation Pane")

    CreateButton = st.sidebar.button("Create Your Database")
    PredictButton = st.sidebar.button("Predict Yoga Pose")
    AddYogaButton = st.sidebar.button("Add Yoga Pose")
    ViewDatabaseButton = st.sidebar.button("View Database Created")
    ViewGraphsButton = st.sidebar.button("View All ML Model Graphs")
    ViewDataFramesButton = st.sidebar.button("View All Data Distributions")

    if(CreateButton):
        st.info("A camera will open if Database doesn't exists.\nPress ' q ' to close camera")
        output = runScript_And_return_TextOutput(creatorDirect)
        if ( output.lower() == "Database Already exists".lower() ) :
            st.info(f"OUTPUT :\n{output}",icon="✌️")
        else :
            st.info(f"OUTPUT :\n{output}",icon="✅")
        
    if(PredictButton):
        output = runScript_And_return_TextOutput(predictorDirect)
        st.info(f"Output :\n{output}❤️")

    if (AddYogaButton):
        runAnotherStreamlitApp(adderStreamlitDirect)

    if(ViewDatabaseButton):
        df = pd.read_csv(databaseDirect)
        st.write(df)
        if(st.button("close database")):
            EmptyBufferFunction()
            
    if(ViewDataFramesButton):
        runAnotherStreamlitApp(dataframesStreamlitDirect)

    if(ViewGraphsButton):
        runAnotherStreamlitApp(resultsStreamlitDirect)
    try: 
        df = pd.read_csv(databaseDirect)
        class_counts = df['class'].value_counts()
        
        plt.figure(figsize=(5, 5))
        class_counts.sort_values().plot(kind='barh', color='skyblue')
        plt.xlabel('Number of Rows')
        plt.ylabel('Class')
        plt.title('Class Distribution in Data')
        st.pyplot(plt)
    except :
        st.write("CSV DATA NOT FOUND\nCREATE ONE")

if __name__== "__main__":
    main()