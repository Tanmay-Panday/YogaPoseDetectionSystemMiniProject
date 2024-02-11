import os
currDirect = os.path.dirname(os.path.realpath(__file__))
befDirect = os.path.dirname(currDirect)
mainDirect = os.path.dirname(befDirect)
modelPath = os.path.join(mainDirect, 'best_model.pkl')

print(os.path.exists(modelPath))

# D:\Yoga Pose with csvData\webApp\best_model.pkl