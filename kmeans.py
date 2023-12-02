import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class Cluster:
	def __init__(self,center,points):
		self.center = center 
		self.points = points
	def addpoint(self,point):
		self.points.append(point)
	def __str__(self):
		points_string = list(map(lambda x: str(x), self.points))
		return "Cluster center = " + str(self.center)+ " Points = ["+str(points_string)+"]"

class K_means:
	def __init__(self,ncluster,file):
		self.n = ncluster
		self.X = np.array(file)
		self.clusters = [0]*ncluster

	def initialize(self):
		np.random.seed(23)
		for x in range(self.n):
			center = 2*(2*np.random.random((self.X.shape[1],))-1)
			self.clusters[x] = Cluster(center,[])

	def distance_x_y(self,x,y):
		return np.sqrt(np.sum((x-y)**2)) 

	def assign(self):
		dist = []
		for y in range(self.X.shape[0]):
			d = []
			current_ex = self.X[y]

			for i in range(self.n):
				distance = self.distance_x_y(current_ex,self.clusters[i].center)
				d.append(distance)
			current_cluster = np.argmin(d)
			self.clusters[current_cluster].addpoint(current_ex)
			dist.append(d[current_cluster])
		return dist

	def updating(self,distances,minimo_local):
		for i in range(self.n):
			points = np.array(self.clusters[i].points)
			if points.shape[0] > 0:
				new = points.mean(axis=0)
				self.clusters[i].center = new 
				if min(distances) > minimo_local:
					self.clusters[i].points = []

	def fit(self):
		self.initialize()
		distances = [100]
		minimo_local = 0.35
		for it in range(0,100):
			distances = self.assign()
			self.updating(distances,minimo_local)
			print(min(distances))
			if min(distances) < minimo_local:
				break

	def predict(self):
		p = []
		for i in range(self.X.shape[0]):
			d = []
			for j in range(self.n):
				d.append(self.distance_x_y(self.X[i],self.clusters[j].center))
			p.append(np.argmin(d))
		return p

	def p(self):
		pred = self.predict()
		print(pred)

		plt.figure(figsize=(12,5))
		plt.subplot(1,2,1)
		plt.scatter(self.X[:,0],self.X[:,1],c = pred)
		for i in self.clusters:
			center = i.center[:2]
			plt.scatter(center[0],center[1],marker = '^',c = 'red')
		plt.xlabel("petal length (cm)")
		plt.ylabel("petal width (cm)")

		plt.subplot(1,2,2)   
		plt.scatter(self.X[:,2],self.X[:,3],c = pred)
		for i in self.clusters:
			center = i.center[2:4]
			plt.scatter(center[0],center[1],marker = '^',c = 'red')
		plt.xlabel("petal length (cm)")
		plt.ylabel("petal width (cm)")
		plt.show() 


	def plot(self,x,y,f):
		pred = self.predict()
		fig = plt.figure(0)
		plt.grid(True)
		plt.scatter(self.X[:,x],self.X[:,y],c = pred)
		for i in self.clusters:
			center = i.center
			plt.scatter(center[0],center[1],marker = '^',c = 'red')
		plt.title("Plot columna {} vs {}".format(f(x),f(y)))
		plt.savefig("plots/plot_columna={}_vs_columna={}.jpg".format(f(x),f(y)), dpi=300)
		plt.clf()
		#plt.show()

	def __str__(self):
		a = ""
		for cluster in self.clusters:
			a += cluster.__str__() + " "
		return a


#X,y = make_blobs(n_samples = 500,n_features = 2,centers = 3,random_state = 23)

iris = pd.read_csv("iris.csv")
X = np.array(iris.drop("species",axis=1))


#fig = plt.figure(0)
#plt.grid(True)
#plt.scatter(X[:,0],X[:,1])
#plt.show()

def f(x):
	if x == 0:
		return "sepal_length"
	elif x == 1:
		return "sepal_width"
	elif x==2:
		return "petal_length"
	else:
		"petal_width"



c = K_means(20,X)
c.fit()
#print(c.clusters[0].__str__())
print(c.clusters[1].__str__())

print(c.predict())
'''
for x in range(4):
	for y in range(4):
		if x!=y:
			c.plot(x,y,f)
#c.plot()'''
