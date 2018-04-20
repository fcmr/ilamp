import numpy as np
import lamp

from sklearn import linear_model

from time import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


GRID_SIZE = 50

# simple plot_embeddings: work on something fancier later
def plot_proj(X, colors):
    plt.axes().set_aspect('equal')
    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.savefig("dataset_projection.png")
    plt.clf()
    #plt.show()

def LoadZeroOneData():
    X = np.load("data/zero_one.npy")
    y = np.load("data/zero_one_label.npy")
    return X, y

def SampleSquare(num_samples, limits):
    pts = []
    for i in range(num_samples):
        x = np.random.uniform(low=limits[0], high=limits[2])
        y = np.random.uniform(low=limits[1], high=limits[3])
        pts.append((x,y))
    return pts

def SparseMap(X, cells, clf):
    gridsz = len(cells)
    smap = np.ones((gridsz, gridsz, 4))

    # Construct the sparse grid based on the list of indices for each cell
    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) == 0:
                continue
            X_sub = X[cells[row][col]]
            labels = clf.predict(X_sub)
            num_zeros = np.sum(labels == 0)
            num_ones = np.sum(labels == 1)
            # we want to quantify if we have more ones than zeros (or more
            # zeros than ones):
            # num_ones == num_zeros -> 0.5
            # all num_ones -> 1.0
            # all num_zeros -> 0.0
            # something like this should work:
            # 0.5 + (num_ones - num_zeros)/2*(num_ones + num_zeros)
            #grid[j, i] = 0.5 + (num_ones - num_zeros)/(2*(num_ones + num_zeros))

            alpha = abs(num_ones - num_zeros)/(num_ones + num_zeros)
            if num_ones > num_zeros:
                smap[row, col] = np.array([1.0, 0.0, 0.0, alpha])
            else:
                smap[row, col] = np.array([0.0, 0.0, 1.0, alpha])
    return smap

def DenseMap(X, proj, smap, cells, clf):
    SAMPLES_PER_CELL = 5
    dmap = np.copy(smap)

    gridsz = len(cells)
    # TODO: projection is normalized [0.0, 1.0], thus min and max are known
    xmin = np.min(proj[:, 0])
    xmax = np.max(proj[:, 0])
    ymin = np.min(proj[:, 1])
    ymax = np.max(proj[:, 1])
    x_intrvls = np.linspace(xmin - 1e-5, xmax + 1e-5, num=gridsz + 1)
    y_intrvls = np.linspace(ymin - 1e-5, ymax + 1e-5, num=gridsz + 1)


    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) >= SAMPLES_PER_CELL:
                continue
            X_sub = [x for x in X[cells[row][col]]]
            num_samples = SAMPLES_PER_CELL - len(cells[row][col])
            limits = [x_intrvls[col], y_intrvls[row], x_intrvls[col + 1], y_intrvls[row + 1]]

            num_samples = SAMPLES_PER_CELL - len(cells[row][col])
            sampled = SampleSquare(num_samples, limits)
            for (x, y) in sampled:
                X_sub.append(lamp.ilamp(X, proj, np.array([x, y])))
            # Compute color for this cell
            labels = clf.predict(X_sub)
            num_zeros = np.sum(labels == 0)
            num_ones = np.sum(labels == 1)
            # we want to quantify if we have more ones than zeros (or more
            # zeros than ones):
            # num_ones == num_zeros -> 0.5
            # all num_ones -> 1.0
            # all num_zeros -> 0.0
            # something like this should work:
            # 0.5 + (num_ones - num_zeros)/2*(num_ones + num_zeros)
            #grid[j, i] = 0.5 + (num_ones - num_zeros)/(2*(num_ones + num_zeros))

            alpha = abs(num_ones - num_zeros)/(num_ones + num_zeros)
            if num_ones > num_zeros:
                dmap[row, col] = np.array([1.0, 0.0, 0.0, alpha])
            else:
                dmap[row, col] = np.array([0.0, 0.0, 1.0, alpha])

    return dmap

def main():
    # 1. loads a 2-class dataset
    print("Loading data...")
    start = time()
    X, y = LoadZeroOneData() 
    print("\tLoading finished: ", time() - start)

    # 2. Trains a linear classifier, just for testing
    print("Training classifier...")
    start = time()
    clf = linear_model.LogisticRegression()
    clf.fit(X, y)
    print("\tTraining finished: ", time() - start)
    #print(clf.score(X, y))

    # 3. Computes LAMP projection:
    print("Projecting points...")
    start = time()
    proj = lamp.lamp2d(X)
    print("\tProjection finished: ", time() - start)
    colors = ['blue', 'red']
    proj_colors = [colors[i] for i in y]

    print("Plotting LAMP projection..")
    start = time()
    plot_proj(proj, proj_colors)
    print("\tPlotting finished: ", time() - start)

    # cells will store the indices of the points that fall inside each cell
    # of the grid
    # Initializes cells
    cells = [[] for i in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        cells[i] = [[] for _ in range(GRID_SIZE)]
    
    tile_size = 1.0/GRID_SIZE
    # Adds point's indices to the corresponding cell
    for idx in range(len(proj)):
        p = proj[idx]
        row = int(abs(p[1] - 1e-5)/tile_size)
        col = int(abs(p[0] - 1e-5)/tile_size)
        cells[row][col].append(idx)


    print("Creating sparse grid new...")
    start = time()
    sparse_map = SparseMap(X, cells, clf)
    print("\tFinished computing sparse grid: ", time() - start)

    print("Plotting sparse grid...")
    start = time()
    tmp_sparse = np.flip(sparse_map, axis=0)
    plt.imshow(tmp_sparse, interpolation='none')
    plt.savefig("sparse_map.png")
    plt.clf()
    print("\tFinished plotting sparse grid: ", time() - start)

    print("Creating dense grid new...")
    start = time()
    dense_map_alt = DenseMap(X, proj, sparse_map, cells, clf)
    print("\tFinished computing dense grid: ", time() - start)

    print("Plotting dense grid...")
    start = time()
    tmp_dense = np.flip(dense_map_alt, axis=0)
    plt.imshow(tmp_dense, interpolation='none')
    plt.savefig("dense_map.png")
    plt.clf()
    print("\tFinished plotting dense grid: ", time() - start)

if __name__ == "__main__":
    main()
