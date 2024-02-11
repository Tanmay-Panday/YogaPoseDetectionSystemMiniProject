import os
currDirect = os.path.dirname(os.path.realpath(__file__))
curvesDirect = os.path.join(currDirect, 'comparison of auc for all roc.jpg')
print(os.path.exists(curvesDirect))
