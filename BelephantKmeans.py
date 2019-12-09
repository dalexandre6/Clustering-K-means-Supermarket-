#import the dataset
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Belephant = pd.read_csv('Belephant.csv')

#We will define credit score as X (independet Variable).
#The credit score is independt from the amount purchased in dollars for this model.
#this means no matter how much moaney the customer expends, the credit score is
#associated to other variables like loyalty, prompt payments, reputation etc...
#Purchases(Y) will be our dependet varible. 


#Now we get rid of the columns we do not need :

#for X we will select all the rows in the column 2 and 3 (index 1 and 2):
X = Belephant.iloc[:, [1,2]].values


#let's call the KMeans algorithm:

kmeans = KMeans(n_clusters=3)

#We use the predict method to show the clusters (in this case we have 3 clusters):
y_kmeans = kmeans.fit_predict(X)

#Let's plot 

#In thw first set, the first column means the number of the cluster(0,1,2)
#since in this case there are just 3 of them. Second column means the index
#in this case all are 0 because is the index AFTER REMOVING THE COLUMNS WE DON'T USE

#In the second set, the furst columns are the number of the cluster and second
#colums are the index (in this case 1) AFTER REMOVING COLUMNOS NOT NEEDED.

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 350, c = 'green', label = 'Premium')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 200, c = 'orange', label = 'To Improve' )
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 30, c = 'red', label = 'To focus' )


#Adding titles and axis labels:

plt.title('Belephant Customers:')
plt.xlabel('Credit Score')
plt.ylabel('Purchase in Millions')
plt.legend()
plt.show()




