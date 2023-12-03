import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from random import uniform
from numpy import asarray

import random
from PIL import Image


class Cluster:
	def __init__(self,center,points):
		self.center = center 
		self.points = points
	def addpoint(self,point):
		self.points.append(point)
	def __str__(self):
		points_string = list(map(lambda x: str(x), self.points))
		return "Cluster center = " + str(self.center)+ " Points = ["+str(len(points_string))+"]"

class K_means:
	def __init__(self,ncluster,file):
		self.n = ncluster
		self.X = np.array(file)
		self.clusters = [0]*ncluster

	def initialize(self):
		for x in range(self.n):
			center = self.X[0]
			self.clusters[x] = Cluster(center,[])

	def distance_x_y(self,x,y):
		potencia=(x-y)**2
		suma = np.sum(potencia)
		raiz = np.sqrt(suma)
		return raiz 

	def assign(self):
		dist = []
		for y in range(self.X.shape[0]):
			d = []
			current_ex = self.X[y]
			for cluster in self.clusters:
				distance = self.distance_x_y(current_ex,cluster.center)
				d.append(distance)
			current_cluster = np.argmin(d)
			self.clusters[current_cluster].addpoint(current_ex)
			dist.append(d[current_cluster])
		return dist

	def updating(self,distances,minimo_local):
		for i in range(self.n):
			points = np.array(self.clusters[i].points)
			if points.shape[0] > 0:
				new = np.mean(points, axis=0)
				self.clusters[i].center = new 
				self.clusters[i].points = []

	def fit(self):
		self.initialize()
		distances = [100]
		minimo_local = 0.01
		cluster = self.clusters
		for it in range(0,5):
			if max(distances) < minimo_local:
				print("out")
				break
			print("assigned")
			distances = self.assign()
			print("update")
			self.updating(distances,minimo_local)

	def pred(self,Y):
		p = []
		for i in range(Y.shape[0]):
			d = []
			for j in range(self.n):
				distance = self.distance_x_y(Y[i],self.clusters[j].center)
				d.append(distance)
			p.append(np.argmin(d))
		return p

	def predict(self):
		p = []
		for i in range(self.X.shape[0]):
			d = []
			for j in range(self.n):
				d.append(self.distance_x_y(self.X[i],self.clusters[j].center))
			p.append(np.argmin(d))
		return p

	def plot(self):
		pred = self.predict()

		plt.figure(figsize=(12,5))
		plt.subplot(1,2,1)
		plt.scatter(self.X[:,0],self.X[:,1],c = pred)
		for i in self.clusters:
			center = i.center
			plt.scatter(center[0],center[1],marker = '^',c = 'red')
		plt.title("Plot Cluster {} sepal length vs sepal width".format(self.n))
		plt.xlabel("sepal length (cm)")
		plt.ylabel("sepal width (cm)")

		plt.subplot(1,2,2)   
		plt.scatter(self.X[:,2],self.X[:,3],c = pred)
		for i in self.clusters:
			center = i.center
			plt.scatter(center[2],center[3],marker = '^',c = 'red')
		plt.xlabel("petal length (cm)")
		plt.ylabel("petal width (cm)")
		plt.title("Plot Cluster {} petal length vs petal width".format(self.n))
		plt.savefig("plots/plot_nclusters={}.jpg".format(self.n), dpi=300)
		plt.clf()

	def __str__(self):
		a = ""
		for cluster in self.clusters:
			a += cluster.__str__() + " "
		return a


iris = pd.read_csv("iris.csv")
X = np.array(iris.drop("species",axis=1))

def f(x):
	if x == 0:
		return "sepal_length"
	elif x == 1:
		return "sepal_width"
	elif x==2:
		return "petal_length"
	else:
		"petal_width"

setosa = X[0: 50 , :]
versicolor = X[50: 100, :]
virginica = X[100:, :]


clusters_test = [] #[1,2,3,4,5,6]
for cluster in clusters_test:
	c = K_means(cluster,X)
	c.fit()
	c.plot()
	pred = c.predict()
	setosa_pred = pred[0:50]
	versicolor_pred = pred[50:100]
	virginica_pred = pred[100:]
	'''
	print("el cluster es ", cluster)
	#podria funcionar para binarios porque los demas no se separan por clase al dividir en clusters
	print(setosa_pred)
	print(versicolor_pred)
	print(virginica_pred)
	
	km = KMeans(n_clusters=cluster, random_state=2,n_init="auto")
	km.fit(X)
	a = km.predict(X, sample_weight='deprecated')
	print(a)'''


def get_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == "RGB":
        channels = 3
    elif image.mode=="RGBA":
    	channels=4
    elif image.mode == "L":
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return pixel_values

data = get_image('imagen.jpg')

def change_color(c):
	if c == 2:
		return (59, 141, 218)
	elif c==1:
		return (59, 218, 83)
	elif c==0:
		return (218, 167, 59)
	return (255, 255, 255)

k=3
#################################
#################################

Z=data[0]
for x in range(1,len(data)):
	Z = np.vstack((Z, data[x]))

a = K_means(k,Z)
a.fit()
prediction = a.predict()

for y in range(len(Z)):
	Z[y] = change_color(prediction[y])

new = []
base = 0
for x in range(data.shape[0]):
	new.append(Z[base:base+data.shape[1]])
	base = base+data.shape[1]
new=np.array(new)

#################################
#################################

im = Image.fromarray(new.astype(np.uint8), 'RGB')
im.save("archivo.png")