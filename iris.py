import mppy_lamp
import lamp
import numpy as np

import matplotlib.pyplot as plt


data = np.loadtxt("iris.txt", delimiter=",")[:, :4]
data = (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))

proj = lamp.lamp2d(data)
plt.scatter(proj[:, 0], proj[:, 1])
plt.show()

new_pt = lamp.ilamp(data, proj, np.array([0.1, 0.1]))
print("new_pt: ", new_pt)


#force_old = lamp.force_old(data)
#plt.scatter(force_old[:, 0], force_old[:, 1])
#plt.show()

#force = mylamp.force_method(data)
#plt.scatter(force[:, 0], force[:, 1])
#plt.show()


# testar projeções
#proj = mylamp.lamp2d(data2, sample_pts)
#plt.scatter(proj[:, 0], proj[:, 1])
#plt.show()
#
#proj2 = lamp.lamp_2d(data)
#plt.scatter(proj2[:, 0], proj2[:, 1])
#plt.show()


# testar calculo do alphas: o meu tá bem diferente da implementação
#sample_pts = np.random.randint(0, len(data), int(np.sqrt(len(data))))
#print(sample_pts)
#
#X = data
#X_s = X[sample_pts]
#
#
#oa = mylamp.other_alpha(X_s, X[0])
#ma = mylamp.my_alpha(X_s, X[0])
#
#print(oa)
#print(ma)

