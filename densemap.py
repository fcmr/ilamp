import numpy as np
import lamp

from sklearn import linear_model
from time import time

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

import sys
import os
from enum import Enum

#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
#from keras.optimizers import RMSprop

from data import *


class Dataset(Enum):
    TOY = 0
    WINE = 1
    SEGMENTATION = 2
    MNIST = 3

class Clf(Enum):
    LOGISTIC_REGRESSION = 0
    SVM = 1
    CNN = 2

# List of parameters
GRID_SIZE = 50 
DATASET = Dataset.TOY
CLF = Clf.LOGISTIC_REGRESSION

NUM_CLASSES = 0
COLORS = [#0 class:
          [], 
          #1 class:
          [],
          #2 classes:
          [[0.0, 0.0, 1.0, 0.5], [1.0, 0.0, 0.0, 0.5]],
          #3 classes:
          [[0.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.5]],
          #4 classes:
          [],
          #5 classes:
          [],
          #6 classes:
          [],
          #7 classes:
          [(0.121, 0.466, 0.705, 0.5),
           (1.0, 0.498, 0.054, 0.5),
           (0.172, 0.627, 0.172, 0.5),
           (0.839, 0.152, 0.156, 0.5),
           (0.580, 0.403, 0.741, 0.5),
           (0.549, 0.337, 0.294, 0.5),
           (0.890, 0.466, 0.760, 0.5)],
          #8 classes:
          [],
          #9 classes:
          [],
          #10 classes:
          [(0.121, 0.466, 0.705, 0.5),
           (1.0, 0.498, 0.054, 0.5),
           (0.172, 0.627, 0.172, 0.5),
           (0.839, 0.152, 0.156, 0.5),
           (0.580, 0.403, 0.741, 0.5),
           (0.549, 0.337, 0.294, 0.5),
           (0.890, 0.466, 0.760, 0.5),
           (0.498, 0.498, 0.498, 0.5),
           (0.737, 0.741, 0.133, 0.5),
           (0.090, 0.745, 0.811, 0.5)]
         ]
CMAP_ORIG = [#0 class:
             [],
             #1 class:
             [],
             #2 classes:
             [0.66, 0.0],
             #3 classes:
             [0.66, 0.33, 0.0],
             #4 classes:
             [],
             #5 classes:
             [],
             #6 classes:
             [],
             #7 classes:
             np.linspace(0.86, 0.0, 7),
             #8 classes:
             [],
             #9 classes:
             [],
             #10 classeS:
             np.linspace(0.86, 0.0, 10),
           ]
CMAP_SYN = [#0 class:
            [],
            #1 class
            [],
            #2 classes:
            [0.558, 0.108],
            #3 classes:
            [0.558, 0.194, 0.108],
            #4 classes:
            [],
            #5 classes:
            [],
            #6 classes:
            [],
            #7 classes:
            np.linspace(0.79, 0.108, 7),
            #8 classes:
            [],
            #9 classes:
            [],
            #10 classes:
            np.linspace(0.79, 0.108, 10)
           ]

np.random.seed(0)

def lerp(a, b, t):
    return (1.0 - t)*a + t*b

# check colorscheme.png for a visual explanation
def SV(c, d):
    if d <= 0.5:
        # a: dark red rgb(0.372, 0.098, 0.145) - hsv(0.972, 1.0, 0.5)
        Sa = 1.0
        Va = 0.5
        # b: dark gray rgb(0.2, 0.2, 0.2) - hsv(0.0, 0.0, 0.2)
        Sb = 0.0
        Vb = 0.2
        # c: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
        Sc = 0.0
        Vc = 0.5
        # d: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
        Sd = 1.0
        Vd = 1.0
        S = lerp(lerp(Sb, Sa, c), lerp(Sc, Sd, c), 2.0*d)
        V = lerp(lerp(Vb, Va, c), lerp(Vc, Vd, c), 2.0*d)
    else:
        # a: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
        Sa = 1.0
        Va = 1.0
        # b: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
        Sb = 0.0
        Vb = 0.5
        # c: light gray rgb(0.8, 0.8, 0.8) - hsv(0.0, 0.0, 0.8)
        Sc = 0.0
        Vc = 0.8
        # d: bright pink rgb(? , ?, ?) - hsv(0.0, 0.2, 1.0)
        Sd = 0.2
        Vd = 1.0
        S = lerp(lerp(Sb, Sa, c), lerp(Sc, Sd, c), 2.0*d - 1.0)
        V = lerp(lerp(Vb, Va, c), lerp(Vc, Vd, c), 2.0*d - 1.0)
    return S, V

def TransferFunc(hsv, k):
    a = 0.0
    b = 1.0
    new_img = np.copy(hsv)
    new_img[:, :, 2] = (a + b*new_img[:, :, 2])**k
    return new_img

# simple plot_embeddings
# TODO: work on something fancier later
def PlotProj(X, colors, path):
    plt.axes().set_aspect('equal')
    plt.scatter(X[:, 0], X[:, 1], color=colors, s=10.0)
    plt.savefig(path)
    plt.clf()
    #plt.show()

def SampleSquare(num_samples, limits):
    pts = []
    for i in range(num_samples):
        x = np.random.uniform(low=limits[0], high=limits[2])
        y = np.random.uniform(low=limits[1], high=limits[3])
        pts.append((x,y))
    return pts

def SparseMap(X, cells, clf, num_per_cell, clf_type, in_shape):
    gridsz = len(cells)
    smap = np.zeros((gridsz, gridsz, 3))

    max_pts = 0
    total_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            total_pts += num_pts
            if num_pts > max_pts:
                max_pts = num_pts

    avg_pts = total_pts/(GRID_SIZE*GRID_SIZE)

    print("\t\tMAX_PTS: ", max_pts)

    cmap = CMAP_ORIG[NUM_CLASSES]

    # Construct the sparse grid based on the list of indices for each cell
    for row in range(gridsz):
        for col in range(gridsz):
            if len(cells[row][col]) < num_per_cell:
                # white cell
                smap[row, col, 2] = 1.0
                #smap[row, col] = np.array([0.0, 0.0, 0.2])
                continue

            num_pts = len(cells[row][col])
            X_sub = X[cells[row][col]]
            if in_shape != None:
                X_sub = np.reshape(X_sub, (num_pts,) + in_shape)

            labels = PredictCLF(X_sub, clf, clf_type)

            counts = np.bincount(labels)
            num_winning = np.max(counts)

            hue = cmap[np.argmax(counts)]

            # certainty c
            c = num_winning/num_pts
            # density d
            d = num_pts/max_pts
            #d = min(0.5*d/avg_pts, 1.0)

            s, v = SV(c, d)
            smap[row, col] = np.array([hue, s, v])
    return smap

def DenseMap(X, proj, smap, cells, clf, num_per_cell, clf_type, in_shape):
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
    total_pts = 0
    for row in range(gridsz):
        for col in range(gridsz):
            num_pts = len(cells[row][col])
            if num_pts < num_per_cell:
                total_pts += num_per_cell
            else:
                total_pts += num_pts
            if num_pts > max_pts:
                max_pts = num_pts

    if max_pts < num_per_cell:
        max_pts = num_per_cell
        print("\t\t\tMAX_PTS == N")

    avg_pts = total_pts/(GRID_SIZE*GRID_SIZE)
    print("avg_pts: ", avg_pts)

    cmap = CMAP_SYN[NUM_CLASSES]
    for row in range(gridsz):
        for col in range(gridsz):
            #if len(cells[row][col]) >= num_per_cell:
            #    continue

            # If there is at least one original sample in this cell, the
            # original colormap is used
            if len(cells[row][col]) != 0:
                cmap = CMAP_ORIG[NUM_CLASSES]
            else:
                cmap = CMAP_SYN[NUM_CLASSES]

            X_sub = [x for x in X[cells[row][col]]]
            num_samples = num_per_cell - len(cells[row][col])

            num_pts = len(cells[row][col])
            if num_samples > 0:
                limits = [x_intrvls[col], y_intrvls[row], x_intrvls[col + 1], y_intrvls[row + 1]]
                sampled = SampleSquare(num_samples, limits)
                for (x, y) in sampled:
                    X_sub.append(lamp.ilamp(X, proj, np.array([x, y])))

                num_pts = num_per_cell 

            if in_shape != None:
                X_sub = np.reshape(X_sub, (num_pts,) + in_shape)
            
            luminance = num_pts/float(max_pts)
            # Compute color for this cell
            labels = PredictCLF(X_sub, clf, clf_type)
            #labels = clf.predict(X)

            counts = np.bincount(labels)
            num_winning = np.max(counts)

            hue = cmap[np.argmax(counts)]
            # certainty c
            c = num_winning/num_pts
            # density d
            d = num_pts/max_pts
            d = min(0.5*d/avg_pts, 1.0)

            s, v = SV(c, d)
            dmap[row, col] = np.array([hue, s, v])
    return dmap

def CNNModel(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', 
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def PredictCLF(X, clf, clf_type):
    y_pred = clf.predict(X)
    if clf_type == Clf.CNN:
        y_pred = np.argmax(y_one_hot, axis=1)
    return y_pred

def main():
    input_shape = None
    one_hot = False
    ctrl_pts = None
    delta = None 

    directory = DATASET.name +  '/'

    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Loading data...")
    start = time()
    # TODO: test split for other datasets
    # TODO: shuffle data
    if DATASET == Dataset.TOY:
        # Toy dataset composed of digits 0 and 1 from MNIST
        X, y = LoadZeroOneData() 
        projection_size = len(X)
        num_samples_list = [1, 5, 10, 15]
    elif DATASET == Dataset.WINE:
        X, y = LoadWineData() 
        projection_size = len(X)
        num_samples_list = [1, 2, 3, 4]
    elif DATASET == Dataset.SEGMENTATION:
        X, y = LoadSegmentationData()
        projection_size = len(X)
        num_samples_list = [1, 2, 4, 6]
        ctrl_pts = 120 
        delta = 1.5
    elif DATASET == Dataset.MNIST:
        X, y = LoadMNISTData()
        X_train = np.copy(X)
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        X_test, y_test = LoadMNISTData('test')
        X_train = X_train.reshape((X.shape[0],) + input_shape)
        X_test = X_test.reshape((X_test.shape[0],) + input_shape )

        y_train = keras.utils.to_categorical(y, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
        one_hot = True
        projection_size = 600
        num_samples_list = [1, 5, 10, 15]

    print("\tLoading finished: ", time() - start)

    # Count the number of classes:
    # Since we use a specific
    global NUM_CLASSES
    NUM_CLASSES = len(np.bincount(y))
    colors = COLORS[NUM_CLASSES]

    # 2. Trains a linear classifier, just for testing
    print("Training classifier...")
    start = time()

    if CLF == Clf.LOGISTIC_REGRESSION:
        clf = linear_model.LogisticRegression()
        clf.fit(X, y)
        print(clf.score(X, y))
    elif CLF == Clf.SVM:
        print("TODO")
        sys.exit(0)
    elif CLF == Clf.CNN:
        clf = CNNModel(input_shape, NUM_CLASSES)
        clf.fit(X_train, y_train, batch_size=128, epochs=14, verbose=1,
                validation_data=(X_test, y_test))
        score = clf.evaluate(X_test, y_test, verbose=0)
        print('Test loss: ', score[0])
        print('Test accuracy: ', score[1])

    print("\tTraining finished: ", time() - start)

    X = X[:projection_size]
    #for ctrl_pts in [35, 49, 75, 100, 120, 150, 200]:
    # 3. Computes LAMP projection:
    print("Projecting points...")
    start = time()
    proj = lamp.lamp2d(X, ctrl_pts, delta)
    print("\tProjection finished: ", time() - start)

    y_pred = PredictCLF(X, clf, CLF)
    #y_pred = PredictCLF(X, clf, CLF)
    proj_colors = [colors[i] for i in y_pred]

    print("Plotting LAMP projection..")
    start = time()
    proj_path = directory + DATASET.name + "_projection.png".format(ctrl_pts)
    PlotProj(proj, proj_colors, proj_path)
    print("\tPlotting finished: ", time() - start)

    #sys.exit(0)

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

    print("Creating sparse grid hsv...")
    start = time()
    sparse_map = SparseMap(X, cells, clf, 1, CLF, input_shape)
    print("\tFinished computing sparse grid hsv: ", time() - start)

    print("Plotting sparse grid...")
    start = time()
    tmp_sparse = np.flip(sparse_map, axis=0)
    rgb_img = hsv_to_rgb(tmp_sparse)
    plt.imshow(rgb_img, interpolation='none')
    fig_title = "{}x{} SparseMap".format(GRID_SIZE, GRID_SIZE)
    plt.title(fig_title)
    #fig_name = "new_hsv_k/sparse_map_hsv_" + str(1) + "_samples.png"
    fig_name = directory + DATASET.name + "_{}x{}_sparse_map.png".format(GRID_SIZE, GRID_SIZE)
    plt.savefig(fig_name)
    plt.clf()
    print("\tFinished plotting sparse grid: ", time() - start)

    #ks = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1]
    ks = [0.4, 0.5, 0.6, 0.7, 0.9, 1.1]
    for k in ks:
        print("Plotting sparse grid...")
        start = time()
        tmp_sparse = np.flip(sparse_map, axis=0)
        # pass V through the transfer function
        tmp_sparse = TransferFunc(tmp_sparse, k)
        rgb_img = hsv_to_rgb(tmp_sparse)
        plt.imshow(rgb_img, interpolation='none')
        fig_title = "{}x{} SparseMap - K = {}".format(GRID_SIZE, GRID_SIZE, k)
        plt.title(fig_title)
        #fig_name = "new_hsv_k/sparse_map_hsv_{}_samples_{}.png".format(1, k)
        fig_name = directory + DATASET.name + "_{}x{}_sparse_map_{}.png".format(GRID_SIZE, GRID_SIZE, k)
        plt.savefig(fig_name)
        plt.clf()
        print("\tFinished plotting sparse grid: ", time() - start)


    for i in num_samples_list:
        print("Creating dense grid hsv " + str(i) + " ...")
        start = time()
        dense_map = DenseMap(X, proj, sparse_map, cells, clf, i, CLF, input_shape)
        print("\tFinished computing dense grid: ", time() - start)

        print("Plotting dense grid...")
        start = time()
        tmp_dense = np.flip(dense_map, axis=0)
        plt.imshow(hsv_to_rgb(tmp_dense), interpolation='none')
        fig_title = "{}x{} DenseMap ({} samples)".format(GRID_SIZE, GRID_SIZE, i)
        plt.title(fig_title)
        fig_name = directory + DATASET.name + "_{}x{}_N_{}_dense_map.png".format(GRID_SIZE, GRID_SIZE, i)
        plt.savefig(fig_name)
        plt.clf()
        print("\tFinished plotting dense grid: ", time() - start)

        for k in ks:
            print("Plotting dense grid...")
            start = time()
            tmp_dense = np.flip(dense_map, axis=0)
            tmp_dense = TransferFunc(tmp_dense, k)
            rgb_img = hsv_to_rgb(tmp_dense)
            plt.imshow(rgb_img, interpolation='none')
            fig_title = "{}x{} DenseMap ({} samples, k = {})".format(GRID_SIZE, GRID_SIZE, i, k)
            plt.title(fig_title)
            fig_name = directory + DATASET.name + "_{}x{}_N_{}_dense_map_{}.png".format(GRID_SIZE, GRID_SIZE, i, k)
            plt.savefig(fig_name)
            plt.clf()
            print("\tFinished plotting dense grid: ", time() - start)

if __name__ == "__main__":
    main()
