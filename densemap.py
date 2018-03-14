
import numpy as np
import lamp

from sklearn import linear_model

from mnist2 import plot_embedding
from time import time

import matplotlib.pyplot as plt

# simple plot_embeddings: work on something fancier later
def plot_proj(X, colors):
    plt.axes().set_aspect('equal')
    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.show()

# 1. loads a 2-class dataset
y = np.load("data/zero_one_label.npy")
X = np.load("data/zero_one.npy")

# 2. Trains a linear classifier, just for testing
clf = linear_model.LogisticRegression()
clf.fit(X, y)

#print(clf.score(X, y))


# 3. Computes LAMP projection:
start = time()
proj = lamp.lamp2d(X)
p_min, p_max = np.min(proj, 0), np.max(proj, 0)
proj = (proj - p_min) / (p_max - p_min)
print("Projection computed - ", time() - start)
#plot_embedding(proj, y, X, "LAMP Projection")
colors = ['blue', 'red']
proj_colors = [colors[i] for i in y]


# limits of the projection
xmin = np.min(proj[:, 0])
xmax = np.max(proj[:, 0])
ymin = np.min(proj[:, 1])
ymax = np.max(proj[:, 1])

grid_size = 15

x_intervals = np.linspace(xmin - 1e-5, xmax + 1e-5, num=grid_size + 1)
y_intervals = np.linspace(ymin - 1e-5, ymax + 1e-5, num=grid_size + 1)

for i in range(grid_size + 1):
    plt.plot([x_intervals[i], x_intervals[i]], [y_intervals[0], y_intervals[-1]], color='black')
    plt.plot([x_intervals[0], x_intervals[-1]], [y_intervals[i], y_intervals[i]], color='black')
plot_proj(proj, proj_colors)

grid = np.zeros((grid_size, grid_size))
dense_map = np.ones((grid_size, grid_size, 4))
num_empty = 0
for i in range(grid_size):
    #x_sub = proj[np.logical_and(proj[:, 0] >= x_intervals[i], proj[:, 1] < x_intervals[i +1])]
    idx_i = np.logical_and(proj[:, 0] >= x_intervals[i], proj[:, 0] < x_intervals[i + 1])
    #for j in range(grid_size):
    for j in range(grid_size - 1, -1, -1):
        idx_j = np.logical_and(proj[:, 1] >= y_intervals[j], proj[:, 1] < y_intervals[j + 1])
        idx = np.logical_and(idx_i, idx_j)
        #sub = x_sub[np.logical_and(x_sub[:, 1] >= y_intervals[j], x_sub[:, 1] < y_intervals[j + 1])]
        proj_sub = proj[idx]
        X_sub = X[idx]

        if proj_sub.shape[0] == 0:
            # no samples in this tile, mark it as -1 
            grid[i, j] = -1 
            num_empty += 1
            # generate sample with iLAMP at the center of the cell
            #x_center = (x_intervals[i] + x_intervals[i + 1])*0.5
            #y_center = (y_intervals[j] + y_intervals[j + 1])*0.5
            #new_sample = lamp.ilamp(X, proj, np.array([x_center, y_center]))
            #label = clf.predict([new_sample])[0]

            #dense_map[i, j] = np.array([0.0, 1.0, 0.0, 1.0])

        else:
            # get all samples that projected to that point and feed them to the
            # classifier
            labels = clf.predict(X_sub)
            num_zeros = np.sum(labels == 0)
            num_ones = np.sum(labels == 1)
            print("cell ", i, ";", j, ": ", num_zeros, ";", num_ones)
            # we want to quantify if we have more ones than zeros (or more
            # zeros than ones):
            # num_ones == num_zeros -> 0.5
            # all num_ones -> 1.0
            # all num_zeros -> 0.0
            # something like this should work:
            # 0.5 + (num_ones - num_zeros)/2*(num_ones + num_zeros)
            grid[i, j] = 0.5 + (num_ones - num_zeros)/(2*(num_ones + num_zeros))

            alpha = abs(num_ones - num_zeros)/(num_ones + num_zeros)
            if num_ones > num_zeros:
                dense_map[i, j] = np.array([1.0, 0.0, 0.0, alpha])
            else:
                dense_map[i, j] = np.array([0.0, 0.0, 1.0, alpha])

print(num_empty)


plt.imshow(np.flip(dense_map, axis=1), interpolation='none')
plt.show()

for i in range(grid_size):
    for j in range(grid_size):
        if grid[i, j] != -1:
            continue
        # generate sample with iLAMP at the center of the cell
        x_center = (x_intervals[i] + x_intervals[i + 1])*0.5
        y_center = (y_intervals[j] + y_intervals[j + 1])*0.5
        new_sample = lamp.ilamp(X, proj, np.array([x_center, y_center]))
        label = clf.predict([new_sample])[0]

        if label == 1:
            dense_map[i, j] = np.array([1.0, 0.65, 0.0, 1.0])
        else:
            dense_map[i, j] = np.array([0.0, 0.65, 1.0, 1.0])


plt.imshow(np.flip(dense_map, axis=1), interpolation='none')
plt.show()



