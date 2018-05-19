import numpy as np
import lamp

from sklearn import linear_model

from time import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

GRID_SIZE = 50
SAMPLES_PER_CELL = 5

np.random.seed(0)

def S(c, d):
    # Sa: dark red rgb(0.372, 0.098, 0.145) - hsv(0.972, 0.74, 0.37)
    Sa = 0.74
    # Sb: dark gray rgb(0.31, 0.31, 0.31) - hsv(0.0, 0.0, 0.31)
    Sb = 0.0
    # Sc: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
    Sc = 0.0
    # Sd: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
    Sd = 1.0
    if d <= 0.5:
        return (Sa*c + Sb*(1.0 - c))*(1 - 2*d) + (Sd*c + Sc*(1.0 - c))*(2*d)
    else:
        return (Sa*c + Sb*(1.0 - c))*(2.0 - 2.0*d) + (Sd*c + Sc*(1.0 - c))*(2*d - 1)

def V(c, d):
    # Sa: dark red rgb(0.372, 0.098, 0.145) - hsv(0.972, 0.74, 0.37)
    Va = 0.37
    # Sb: dark gray rgb(0.31, 0.31, 0.31) - hsv(0.0, 0.0, 0.31)
    Vb = 0.31
    # Sc: half gray rgb(0.7, 0.7, 0.7) - hsv(0.0, 0.0, 0.5)
    Vc = 0.5
    # Sd: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
    Vd = 1.0
    if d <= 0.5:
        return (Va*c + Vb*(1.0 - c))*(1 - 2*d) + (Vd*c + Vc*(1.0 - c))*(2*d)
    else:
        return (Va*c + Vb*(1.0 - c))*(2.0 - 2.0*d) + (Vd*c + Vc*(1.0 - c))*(2*d - 1)

# simple plot_embeddings: work on something fancier later
def PlotProj(X, colors):
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

def SparseMap(X, cells, clf, num_per_cell=1, k=0.3):
    gridsz = len(cells)
    smap = np.ones((gridsz, gridsz, 4))

    max_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts > max_pts:
                max_pts = num_pts

    a = 0.1
    b = 0.9

    cmap = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])]

    # Construct the sparse grid based on the list of indices for each cell
    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) < num_per_cell:
                continue
            X_sub = X[cells[row][col]]
            labels = clf.predict(X_sub)

            counts = np.bincount(labels)
            num_winning = np.max(counts)
            num_pts = len(cells[row][col])
            alpha = max(2*num_winning - num_pts, 0)/num_pts

            c = cmap[np.argmax(counts)]

            #num_zeros = np.sum(labels == 0)
            #num_ones = np.sum(labels == 1)

            s = num_pts/float(max_pts)
            c = c*((a + b*s)**k)
            smap[row, col] = np.array([c[0], c[1], c[2], alpha])
            #if num_ones > num_zeros:
            #    c = np.array([1.0, 0.0, 0.0])
            #    alpha = num_ones/(num_ones + num_zeros)
            #    #smap[row, col] = np.array([1.0, 0.0, 0.0, alpha])
            #else:
            #    c = np.array([0.0, 0.0, 1.0])
            #    alpha = num_zeros/(num_ones + num_zeros)
            #    #smap[row, col] = np.array([0.0, 0.0, 1.0, alpha])

    return smap

def SparseMapHSV(X, cells, clf):
    gridsz = len(cells)
    smap = np.zeros((gridsz, gridsz, 3))

    max_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts > max_pts:
                max_pts = num_pts

    # Construct the sparse grid based on the list of indices for each cell
    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) == 0:
                # white cell
                smap[row, col, 2] = 1.0
                continue

            num_pts = len(cells[row][col])
            saturation = num_pts/float(max_pts)
            X_sub = X[cells[row][col]]
            labels = clf.predict(X_sub)
            num_zeros = np.sum(labels == 0)
            num_ones = np.sum(labels == 1)

            if num_ones > num_zeros:
                hue = 0.0
                luminance = num_ones/(num_ones + num_zeros)
                smap[row, col] = np.array([hue, saturation, luminance])
            else:
                hue = 0.65
                luminance = num_zeros/(num_ones + num_zeros)
                smap[row, col] = np.array([hue, saturation, luminance])
    return smap

def SparseMapHSVNew(X, cells, clf, num_per_cell=1):
    gridsz = len(cells)
    smap = np.zeros((gridsz, gridsz, 3))

    max_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts > max_pts:
                max_pts = num_pts

    # color map: hue values for the two-class problem
    cmap = [0.65, 0.0]

    # Construct the sparse grid based on the list of indices for each cell
    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) < num_per_cell:
                # white cell
                smap[row, col, 2] = 1.0
                continue

            X_sub = X[cells[row][col]]
            labels = clf.predict(X_sub)

            counts = np.bincount(labels)
            num_winning = np.max(counts)
            num_pts = len(cells[row][col])

            hue = cmap[np.argmax(counts)]

            # certainty c
            c = num_winning/num_pts
            # density d
            d = num_pts/max_pts

            s = S(c, d)
            v = V(c, d)
            smap[row, col] = np.array([hue, s, v])


            #saturation = num_winning/num_pts
            #hsv = np.array([hue, saturation, 1.0])
            #hsv_white = np.array([0.0, 0.0, 1.0])
            #smap[row, col] = (1.0 - d)*hsv_white + d*hsv

            #if num_ones > num_zeros:
            #    hue = 0.0
            #    luminance = num_ones/(num_ones + num_zeros)
            #    smap[row, col] = np.array([hue, saturation, luminance])
            #else:
            #    hue = 0.65
            #    luminance = num_zeros/(num_ones + num_zeros)
            #    smap[row, col] = np.array([hue, saturation, luminance])
    return smap



def DenseMap(X, proj, smap, cells, clf, num_per_cell, k=0.3):
    dmap = np.copy(smap)

    gridsz = len(cells)
    # TODO: projection is normalized [0.0, 1.0], thus min and max are known
    xmin = np.min(proj[:, 0])
    xmax = np.max(proj[:, 0])
    ymin = np.min(proj[:, 1])
    ymax = np.max(proj[:, 1])
    x_intrvls = np.linspace(xmin - 1e-5, xmax + 1e-5, num=gridsz + 1)
    y_intrvls = np.linspace(ymin - 1e-5, ymax + 1e-5, num=gridsz + 1)

    max_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts > max_pts:
                max_pts = num_pts

    a = 0.1
    b = 0.9

    # dense map needs a secondary color map to color cells with synthetic 
    # samples different than the first colormap
    cmap = [np.array([0.0, 0.65, 1.0]), np.array([1.0, 0.65, 0.0])]

    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) >= num_per_cell:
                continue
            X_sub = [x for x in X[cells[row][col]]]
            num_samples = num_per_cell - len(cells[row][col])
            limits = [x_intrvls[col], y_intrvls[row], x_intrvls[col + 1], y_intrvls[row + 1]]

            num_samples = num_per_cell - len(cells[row][col])
            sampled = SampleSquare(num_samples, limits)
            for (x, y) in sampled:
                X_sub.append(lamp.ilamp(X, proj, np.array([x, y])))
            # Compute color for this cell
            labels = clf.predict(X_sub)
            
            counts = np.bincount(labels)
            num_winning = np.max(counts)
            # the number of samples in this cell is num_per_cell now that
            # new samples have been created
            num_pts = num_per_cell 
            alpha = max(2*num_winning - num_pts, 0)/num_pts

            c = cmap[np.argmax(counts)]

            #num_zeros = np.sum(labels == 0)
            #num_ones = np.sum(labels == 1)

            s = num_pts/float(max_pts)
            c = c*((a + b*s)**k)
            dmap[row, col] = np.array([c[0], c[1], c[2], alpha])

            #alpha = abs(num_ones - num_zeros)/(num_ones + num_zeros)
            #b_comp = 0.0
            #if num_per_cell == 1:
            #    b_comp = 0.65
            #    alpha = 1.0
            #b_comp = 0.65
            #if num_ones > num_zeros:
            #    #dmap[row, col] = np.array([1.0, 0.65, 0.0, alpha])
            #    #dmap[row, col] = np.array([1.0, 0.0, 0.0, alpha])
            #    c = np.array([1.0, b_comp, 0.0])
            #    #dmap[row, col] = np.array([1.0, b_comp, 0.0, alpha])
            #    alpha = num_ones/(num_ones + num_zeros)
            #else:
            #    #dmap[row, col] = np.array([0.0, 0.0, 1.0, alpha])
            #    #dmap[row, col] = np.array([0.0, b_comp, 1.0, alpha])
            #    c = np.array([0.0, b_comp, 1.0])
            #    alpha = num_zeros/(num_ones + num_zeros)


    return dmap

def DenseMapHSV(X, proj, smap, cells, clf, num_per_cell):
    dmap = np.copy(smap)

    gridsz = len(cells)
    # TODO: projection is normalized [0.0, 1.0], thus min and max are known
    xmin = np.min(proj[:, 0])
    xmax = np.max(proj[:, 0])
    ymin = np.min(proj[:, 1])
    ymax = np.max(proj[:, 1])
    x_intrvls = np.linspace(xmin - 1e-5, xmax + 1e-5, num=gridsz + 1)
    y_intrvls = np.linspace(ymin - 1e-5, ymax + 1e-5, num=gridsz + 1)

    max_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts > max_pts:
                max_pts = num_pts


    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) >= num_per_cell:
                continue
            X_sub = [x for x in X[cells[row][col]]]
            num_samples = num_per_cell - len(cells[row][col])
            limits = [x_intrvls[col], y_intrvls[row], x_intrvls[col + 1], y_intrvls[row + 1]]

            num_samples = num_per_cell - len(cells[row][col])
            sampled = SampleSquare(num_samples, limits)
            for (x, y) in sampled:
                X_sub.append(lamp.ilamp(X, proj, np.array([x, y])))
            
            num_pts = num_per_cell 
            luminance = num_pts/float(max_pts)
            #luminance = 1.0 - 2.0/(np.exp(2*luminance) + 1)
            # Compute color for this cell
            labels = clf.predict(X_sub)
            num_zeros = np.sum(labels == 0)
            num_ones = np.sum(labels == 1)

            if num_ones > num_zeros:
                hue = 0.0
                saturation = num_ones/(num_ones + num_zeros)
                #dmap[row, col] = np.array([hue, luminance, saturation])
                dmap[row, col] = np.array([hue, saturation, luminance])
            else:
                hue = 0.65
                saturation = num_zeros/(num_ones + num_zeros)
                #dmap[row, col] = np.array([hue, luminance, saturation])
                dmap[row, col] = np.array([hue, saturation, luminance])

    return dmap


def DenseMapHSVNew(X, proj, smap, cells, clf, num_per_cell=1):
    dmap = np.copy(smap)

    gridsz = len(cells)
    # TODO: projection is normalized [0.0, 1.0], thus min and max are known
    xmin = np.min(proj[:, 0])
    xmax = np.max(proj[:, 0])
    ymin = np.min(proj[:, 1])
    ymax = np.max(proj[:, 1])
    x_intrvls = np.linspace(xmin - 1e-5, xmax + 1e-5, num=gridsz + 1)
    y_intrvls = np.linspace(ymin - 1e-5, ymax + 1e-5, num=gridsz + 1)

    max_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts > max_pts:
                max_pts = num_pts


    # color map: hue values for the two-class problem
    cmap = [0.558, 0.108]
    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) >= num_per_cell:
                continue
            X_sub = [x for x in X[cells[row][col]]]
            limits = [x_intrvls[col], y_intrvls[row], x_intrvls[col + 1], y_intrvls[row + 1]]

            num_samples = num_per_cell - len(cells[row][col])
            sampled = SampleSquare(num_samples, limits)
            for (x, y) in sampled:
                X_sub.append(lamp.ilamp(X, proj, np.array([x, y])))
            
            num_pts = num_per_cell 
            luminance = num_pts/float(max_pts)
            #luminance = 1.0 - 2.0/(np.exp(2*luminance) + 1)
            # Compute color for this cell
            labels = clf.predict(X_sub)

            counts = np.bincount(labels)
            num_winning = np.max(counts)

            hue = cmap[np.argmax(counts)]
            # certainty c
            c = num_winning/num_pts
            # density d
            d = num_pts/max_pts

            s = S(c, d)
            v = V(c, d)
            dmap[row, col] = np.array([hue, s, v])


            #saturation = num_winning/num_pts
            #s = num_pts/max_pts
            #hsv = np.array([hue, saturation, 1.0])
            #hsv_white = np.array([0.0, 0.0, 1.0])
            #dmap[row, col] = (1.0 - s)*hsv_white + s*hsv

    return dmap

def normalize(a):
    #amin = a.min()
    #amax = a.max()
    return 0.5 + (a*0.5)

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
    y_pred = clf.predict(X)
    colors = ['blue', 'red']
    proj_colors = [colors[i] for i in y_pred]

    print("Plotting LAMP projection..")
    start = time()
    PlotProj(proj, proj_colors)
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

    #for k in [0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.4, 0.5, 0.6, 0.7]:
    #    for i in [1, 5, 10, 15]:
    #        print("Creating sparse grid...")
    #        start = time()
    #        sparse_map = SparseMap(X, cells, clf, i, k)
    #        print("\tFinished computing sparse grid: ", time() - start)

    #        # checks which cells from the sparse map are white because of the alpha
    #        #debug_sm = np.ones((GRID_SIZE, GRID_SIZE, 3))
    #        #for i in range(GRID_SIZE):
    #        #    for j in range(GRID_SIZE):
    #        #        if sparse_map[i, j, 3] == 0.0:
    #        #            debug_sm[i, j] = np.array([0.0, 0.0, 0.0])

    #        print("Plotting sparse grid...")
    #        start = time()
    #        tmp_sparse = np.flip(sparse_map, axis=0)
    #        plt.imshow(tmp_sparse, interpolation='none')
    #        plt.savefig("k2/sparse_map_" + str(i) + "_samples_" + str(k) + ".png")
    #        plt.clf()
    #        print("\tFinished plotting sparse grid: ", time() - start)

    #        #print("Plotting debug sparse grid...")
    #        #start = time()
    #        #tmp_sparse = np.flip(debug_sm, axis=0)
    #        #plt.imshow(tmp_sparse, interpolation='none')
    #        #plt.savefig("sparse_map_debug.png")
    #        #plt.clf()
    #        #print("\tFinished plotting sparse grid: ", time() - start)


    #    # FIXME: compute dense maps only once and plot them differently
    #    #for i in [1, 5, 10, 15]:
    #        print("Creating dense grid " + str(i) + "...")
    #        start = time()
    #        dense_map = DenseMap(X, proj, sparse_map, cells, clf, i, k)
    #        print("\tFinished computing dense grid: ", time() - start)

    #        print("Plotting dense grid...")
    #        start = time()
    #        tmp_dense = np.flip(dense_map, axis=0)
    #        plt.imshow(tmp_dense, interpolation='none')
    #        plt.savefig("k2/dense_map_" + str(i)+ "_samples_" + str(k) + ".png")
    #        plt.clf()
    #        print("\tFinished plotting dense grid: ", time() - start)

    
    for i in [1, 5, 10, 15]:
        print("Creating sparse grid hsv...")
        start = time()
        sparse_map = SparseMapHSVNew(X, cells, clf, i)
        print("\tFinished computing sparse grid hsv: ", time() - start)

        print("Plotting sparse grid...")
        start = time()
        tmp_sparse = np.flip(sparse_map, axis=0)
        plt.imshow(hsv_to_rgb(tmp_sparse), interpolation='none')
        plt.savefig("new_hsv/sparse_map_hsv_" + str(i) + "_samples.png")
        plt.clf()
        print("\tFinished plotting sparse grid: ", time() - start)

        print("Creating dense grid hsv " + str(i) + " ...")
        start = time()
        dense_map = DenseMapHSVNew(X, proj, sparse_map, cells, clf, i)
        #dense_map[:, :, 2] = normalize(dense_map[:, :, 2])
        #dense_map[:, :, 1] = normalize(dense_map[:, :, 1])
        print("\tFinished computing dense grid: ", time() - start)

        print("Plotting dense grid...")
        start = time()
        tmp_dense = np.flip(dense_map, axis=0)
        plt.imshow(hsv_to_rgb(tmp_dense), interpolation='none')
        plt.savefig("new_hsv/dense_map_hsv_" + str(i) + "_samples.png")
        plt.clf()
        print("\tFinished plotting dense grid: ", time() - start)

if __name__ == "__main__":
    main()
