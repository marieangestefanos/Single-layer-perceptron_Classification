import numpy as np
import matplotlib.pyplot as plt
rd = np.random.randn

def new_data(N, mA, mB, sigmaA, sigmaB, seed, rule, type_data):
    np.random.seed(seed)
    classA = np.zeros((2, N))
    classB = np.zeros((2, N))
    if type_data=='linear':
        classA[0, :] = mA[0] + sigmaA * rd(1, N)
        classA[1, :] = mA[1] + sigmaA * rd(1, N)
        classB[0, :] = mB[0] + sigmaB * rd(1, N)
        classB[1, :] = mB[1] + sigmaB * rd(1, N)

        plt.figure()
        plt.scatter(classA[0, :], classA[1, :], c='red')
        plt.scatter(classB[0, :], classB[1, :], c='blue')
        plt.grid(True)
        X = np.concatenate((classA, classB), axis=1)
        X = np.concatenate((X, np.ones((1, N + N))))

        T = np.ones((1, N + N))

        if rule == 'perceptron':
            T[0, N:] = 0
        elif rule == 'delta':
            T[0, N:] = -1
        else:
            print('Enter a correct learning rule: perceptron or delta.')

        shuffler = np.random.permutation(N + N)
        X = X[:, shuffler]
        T = T[0, shuffler]

    if type_data=='nonlinear':

        classA[0, :] = [rd(1,round(0.5*N)) * sigmaA - mA[0] ,rd(1,round(0.5*N)) * sigmaA + mA[0]]
        classA[1, :] = rd(1,N) * sigmaA + mA[1]
        classB[0, :] = rd(1,N) * sigmaB + mB[0]
        classB[1, :] = rd(1,N) * sigmaB + mB[1]

        plt.figure()
        plt.scatter(classA[0, :], classA[1, :], c='red')
        plt.scatter(classB[0, :], classB[1, :], c='blue')

        X = np.concatenate((classA, classB), axis=1)
        X = np.concatenate((X, np.ones((1, N + N))))

        T = np.ones((1, N + N))

        if rule == 'perceptron':
            T[0, N:] = 0
        elif rule == 'delta':
            T[0, N:] = -1
        else:
            print('Enter a correct learning rule: \'perceptron\' or \'delta\'.')

        shuffler = np.random.permutation(N + N)
        X = X[:, shuffler]
        T = T[0, shuffler]
    

    return X, T


def plot_boundary(X, T, W, epoch, pt_idx):
    x1 = np.linspace(-4, 4, 100)
    x2 = np.linspace(-2, 2, 100)
    plt.figure()
    plt.axis([-4, 4, -2, 2])

    _, N = np.shape(X)
    for i in range(N):
        if T[i] == 1:
            plt.scatter(X[0, i], X[1, i], c='red')
        else:
            plt.scatter(X[0, i], X[1, i], c='blue')
    y = (-W[0, 0]*x1 - W[0, 2]*x2)/W[0, 1]
    plt.plot(x1, y)
    plt.title(f"epoch={epoch}, pt_idx={pt_idx}")
    plt.grid(True)
    plt.show()


def perceptron_rule(X, T, W, i, thresh):
    y_prime = W @ X[:, i]
    if y_prime > thresh:
        e = T[i] - 1
    else:
        e = T[i]
    return e


def delta_rule(X, T, W, i, thresh):
    return T[i] - W @ X[:, i]


def perceptron(X, T, rule, mode, epochs, thresh, eta):

    W = np.random.randn(1, 3)

    if rule == 'perceptron':

        print("Perceptron learning")

        #BATCH MODE
        if mode == 'batch':
            print("Mode: batch")
            for epoch in range(epochs):
                accW = 0
                for i in range(N + N):
                    e = perceptron_rule(X, T, W, i, thresh)
                    deltaW = eta * e * X[:, i].T
                    accW += deltaW
                W = W + accW
                plot_boundary(X, T, W, epoch, i)

        ## ONLINE MODE
        elif mode == 'online':
            print("Mode: online")
            for epoch in range(epochs):
                for i in range(N + N):
                    e = perceptron_rule(X, T, W, i, thresh)
                    deltaW = eta * e * X[:, i].T
                    W = W + deltaW
                plot_boundary(X, T, W, epoch, i)
        else:
            print('Enter a correct mode: online or batch.')

    elif rule == 'delta':

        print("Delta rule.")

        # BATCH MODE
        if mode == 'batch':
            print("Mode: batch")
            for epoch in range(epochs):
                deltaW = - eta * (W@X - T) @ X.T
                W += deltaW
                plot_boundary(X, T, W, epoch, -1)

        ## ONLINE MODE
        elif mode == 'online':
            print("Mode: online")
            for epoch in range(epochs):
                for i in range(N + N):
                    e = delta_rule(X, T, W, i, thresh)
                    deltaW = eta * e * X[:, i].T
                    W = W + deltaW
                plot_boundary(X, T, W, epoch, i)
        else:
            print('Enter a correct mode: online or batch.')

    else:
        print('Enter a correct learning rule: perceptron or delta.')






