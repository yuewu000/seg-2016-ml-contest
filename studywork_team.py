
"""
Lithology Classify modeling for rock physicis prediction
-Machine learning method: 
 grid search based on SVM.SVC (support vector machine classification modeling)
Created on Tue Jan 07 17:35:11 2017

@author: Yue Wu; Yong Chang; Hang Ji; Fangwei Fu
 """

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt

filein = "./training_data.csv"
datatr = pd.read_csv(filein)
print ("Names of all the wells are:\n")
print (set(datatr["Well Name"]))

"""
Names of all the 8 wells are displayed as above:
Well SHANKLE is reserved as test well. 
"""

#prepare data
test_name = 'SHANKLE'
test_well = datatr[datatr['Well Name'] == test_name]
test2_well = datatr[datatr['Well Name'] == test_name]
test3_well = datatr[datatr['Well Name'] == test_name]
data = datatr[datatr['Well Name'] != test_name]

characters = ['GR','ILD_log10','DeltaPHI','PHIND','PE','NM_M','RELPOS']
character_vectors = data[characters]
facies_label = data['Facies']

"""
Plots to study the data sets: 
    * scatterplot diagram with histograms;
    * sample joint hexin plot with histograms for each attribute
      Here we take the example of “hexbin” plot, which shows 
      the counts of observations that fall within hexagonal bins;
    * sample 3D scatter plot
"""
#advanced plot with seaborn
sns.set(style='whitegrid')
sns.set_color_codes('pastel')

#scatterplot diagram    
sns.pairplot(character_vectors[['GR','ILD_log10','DeltaPHI','PHIND','PE']])

#joint hexin plot with histograms for both
sns.jointplot(x="ILD_log10",y="PE",data=character_vectors,\
              kind="hex", space=0, color="g")
#3D scatter
from mpl_toolkits.mplot3d import Axes3D
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = pylab.figure()
ax = Axes3D(fig)

ax.scatter(character_vectors['GR'], character_vectors['DeltaPHI'],character_vectors['PHIND'],color='red')
ax.set_xlabel('PE')
ax.set_ylabel('DeltaPHI')
ax.set_zlabel('PHIND')
plt.show()

#standarlization, scale the data features 
scl = StandardScaler().fit(character_vectors )
scld_characters = scl.transform(character_vectors )

#split to training dat and crossvalidation data
xtr, xcv, ytr, ycv = \
    train_test_split(scld_characters, facies_label, test_size=0.05, random_state=42)

# Grid search method: defien the grid for the seach
# Here C_range and gamma_range are scaned
   
C_range = np.arange(0.601,1.601,0.2)
gamma_range = np.arange(0.0001,0.1201,0.02)

param_grid = [{"gamma":gamma_range.tolist(),"C":C_range.tolist(),\
              "class_weight":['balanced',None]}]

# Create SVM model
svc1 = svm.SVC(verbose=False)

grid = GridSearchCV(svc1, param_grid, cv=5, scoring="precision", verbose=1)

grid.fit(xtr,ytr)
print("The best classifier is:", grid.best_estimator_)
 
# Extract facies labels and scale the test data features 
# the same as the scaling on the training data
y_test = test_well['Facies']
well_characters = test_well.drop(['Facies',
                                 'Formation',
                                 'Well Name',
                                 'Depth'],
                                 axis=1)
x_test = scl.transform(well_characters)

# predict and evaluate the model using the test data
y_prd = grid.predict(x_test)

test_well['Prediction'] = y_prd
target_names = ['SS', 'CSiS', 'FSiS','SiSH','MS','WS','D','PS','BS']
print(classification_report(y_test, y_prd, target_names=target_names))

