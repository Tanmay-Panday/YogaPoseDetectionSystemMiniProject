import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

currDirect = os.path.dirname(os.path.realpath(__file__))
databaseDirect = os.path.join(currDirect, 'PythonCodes and Database', 'Database', 'coordinates.csv')
train_path = os.path.join(currDirect, 'PythonCodes and Database', 'Database', 'train.csv')
test_path = os.path.join(currDirect, 'PythonCodes and Database', 'Database', 'test.csv')

def EmptyBufferFunction():
    pass

def viewDistributiion(path):
    try: 
        df = pd.read_csv(path)
        class_counts = df['class'].value_counts()
        
        plt.figure(figsize=(5, 5))
        class_counts.sort_values().plot(kind='barh', color='skyblue')
        plt.xlabel('Number of Rows')
        plt.ylabel('Class')
        plt.title('Class Distribution in DataFrame')
        st.pyplot(plt)
    except :
        st.write("CSV DATA NOT FOUND\nCREATE ONE")

def main():
    st.title("VIEW THE DATAFRAMES USED")
    clButton = st.button("Close Distribution Viewer")
    ogButton = st.button("Original Data")
    trnButton = st.button("Train Data")
    tstButton = st.button("Test Data")

    if(clButton):
        os._exit(0)

    if(ogButton):
        if(st.button("close")):
            EmptyBufferFunction()
        st.info("Original Database Distribution")
        viewDistributiion(databaseDirect)


    if(trnButton):
        if(st.button("close")):
            EmptyBufferFunction()
        st.info("Training Database Distribution")
        viewDistributiion(train_path)


    if(tstButton):
        if(st.button("close")):
            EmptyBufferFunction()
        st.info("Testing Database Distribution")
        viewDistributiion(test_path)


if __name__== "__main__":
    main()
