import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn as rd


def new_data(mA, mB, sigmaA, sigmaB, n, data_type, rng):

    if data_type == "linear":
        classA = rng.standard_normal((2, n))*sigmaA + np.repeat(mA, n, axis=1)
        classB = rng.standard_normal((2, n))*sigmaB + np.repeat(mB, n, axis=1)

    elif data_type == "nonlinear":
        N = n+n

        #First coord classA
        classA00 = rng.standard_normal((1, int(n / 2))) * sigmaA + mA[0]
        classA01 = rng.standard_normal((1, int(n / 2))) * sigmaA - mA[0]
        classA0 = np.concatenate((classA00, classA01), axis=1) #left and right clusters
        #Second coord classA
        classA1 = rng.standard_normal((1, n))*sigmaA + mA[1]

        classA = np.concatenate((classA0, classA1), axis=0)
        classB = rng.standard_normal((2, n))*sigmaB + np.repeat(mB, n, axis=1)


    X2D = np.concatenate((classA, classB), axis = 1)
    X = np.concatenate((X2D, np.ones((1, n+n))), axis = 0)

    T = np.concatenate((np.ones(n), -np.ones(n)))

    shuffler = rng.permutation(n+n)
    X = X[:, shuffler]
    T = T[shuffler]

    return X, T


def plot_data(X, T):
    plt.scatter(X[0, :], X[1, :], c=T)
    plt.xlim([-1.7, 1.7])
    plt.ylim([-1.7, 1.7])
    plt.grid()


def plot_boundaries(W):
    x = np.linspace(-1, 3, 100)
    y = - ( W[2] + W[0]*x ) / W[1]
    # y = - ( W[0]*x ) / W[1]

    plt.plot(x, y, color='red')


def subsample(n, scenario):
    N = n + n

    # First coord classA
    classA00 = rng.standard_normal((1, int(n / 2))) * sigmaA + mA[0]
    classA01 = rng.standard_normal((1, int(n / 2))) * sigmaA - mA[0]
    classA0 = np.concatenate((classA00, classA01), axis=1)  # left and right clusters
    # Second coord classA
    classA1 = rng.standard_normal((1, n)) * sigmaA + mA[1]

    if scenario == 1:
        m = 75
        p = 75
        classA = np.concatenate((classA0, classA1), axis=0)
        classA = rng.choice(classA.T, size=m).T

        classB = rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1)
        classB = rng.choice(classB.T, size=p).T

        print(classA.shape)
        print(classB.shape)

    elif scenario == 2:
        m = 50

        classA = np.concatenate((classA0, classA1), axis=0)
        classA = rng.choice(classA.T, size=m).T

        classB = rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1)
        p = classB.shape[1]

        print(classA.shape)
        print(classB.shape)

    elif scenario == 3:
        p = 50

        classA = np.concatenate((classA0, classA1), axis=0)
        m = classA.shape[1]

        classB = rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1)
        classB = rng.choice(classB.T, size=p).T

        print(classA.shape)
        print(classB.shape)

    elif scenario == 4:
        m0 = 10
        m1 = 40

        classA00 = rng.standard_normal((1, int(n / 2))) * sigmaA + mA[0]
        classA00 = rng.choice(classA00.T, size=m0).T
        classA01 = rng.standard_normal((1, int(n / 2))) * sigmaA - mA[0]
        classA01 = rng.choice(classA01.T, size=m1).T

        classA0 = np.concatenate((classA00, classA01), axis=1)  # left and right clusters
        # Second coord classA
        classA1 = rng.standard_normal((1, m0+m1)) * sigmaA + mA[1]

        classA = np.concatenate((classA0, classA1), axis=0)
        m = classA.shape[1]

        classB = rng.standard_normal((2, n)) * sigmaB + np.repeat(mB, n, axis=1)

        p = classB.shape[1]


        print(classA.shape)
        print(classB.shape)

    else:
        print("Please enter a correct scenario : 1, 2, 3 or 4.")



    X2D = np.concatenate((classA, classB), axis=1)
    X = np.concatenate((X2D, np.ones((1, m+p))), axis=0)

    T = np.concatenate((np.ones(m), -np.ones(p)))

    shuffler = rng.permutation(m+p)
    X = X[:, shuffler]
    T = T[shuffler]
    return X, T


def missclassified_rate(X, T, W):
    predictions = np.where(W@X>0, 1, 0)
    T_01 = np.where(T==1, 1, 0)
    accuracy = np.zeros((2, 2))
    for i in range(X.shape[1]):
        target = T_01[i]
        if predictions[i] == target:
            accuracy[target, target] += 1
        elif predictions[i] > target:
            accuracy[0, 1] += 1
        else:
            accuracy[1, 0] += 1
    accuracy[0, :] *= 100/(accuracy[0, 0] + accuracy[0, 1])
    accuracy[1, :] *= 100/(accuracy[1, 0] + accuracy[1, 1])

    return accuracy


n = 100
mA = np.array([[1.0], [0.3]])
mB = np.array([[-1.0], [-0.1]])
sigmaA = 0.2
sigmaB = 0.3

data_type = "linear"
#data_type = "nonlinear"
rng = np.random.default_rng()

X, T = new_data(mA, mB, sigmaA, sigmaB, n, data_type, rng)
# X, T = subsample(n, 1)
#X, T = subsample(n, 2)
# X, T = subsample(n, 3)
# X, T = subsample(n, 4)

epochs = 20
eta = 1e-3

W = rng.random((3))
N = X.shape[1]


# #PERCEPTRON ONLINE
# plt.ion()
#
# mse = []
# accuracyA = []
# accuracyB = []
# for epoch in range(epochs):
#     for i in range(N):
#         y_prime = W@X[:, i]
#         if y_prime > 0:
#             y = 1
#         else:
#             y = 0
#         e = int(T[i]>0) - y
#         deltaW = eta * e * X[:, i]
#         W += deltaW
#
#     mse.append(np.mean((W@X-T)**2))
#     plt.clf()
#     plot_data(X, T)
#     plot_boundaries(W)
#     plt.title(f"epoch:{epoch}")
#     plt.pause(0.1)
#
#     print(f"EPOCH={epoch}, accuracy:")
#     accuracy = missclassified_rate(X, T, W)
#     print(accuracy)
#     accuracyA.append(accuracy[1, 1])
#     accuracyB.append(accuracy[0, 0])
#
# plt.ioff()
#
# plt.figure()
# plt.plot(range(epochs), mse)
# plt.title("MSE")
# plt.show()
#
# plt.figure()
# plt.plot(range(epochs), accuracyA)
# plt.plot(range(epochs), accuracyB)
# plt.title("Accuracy")
# plt.show()




# #DELTA ONLINE
# plt.ion()
#
# mse = []
#
# for epoch in range(epochs):
#     for i in range(N):
#         e = T[i] - W@X[:, i]
#         deltaW = eta * e * X[:, i]
#         W += deltaW
#     mse.append(np.mean((W@X-T)**2))
#     plt.clf()
#     plot_data(X, T)
#     plot_boundaries(W)
#     plt.title(f"epoch:{epoch}")
#     plt.pause(0.2)
#
# plt.ioff()
#
# plt.figure()
# plt.plot(range(epochs), mse)
# plt.title("MSE")
# plt.show()
#
# plt.figure()
# plt.plot(range(epochs), accuracyA)
# plt.plot(range(epochs), accuracyB)
# plt.title("Accuracy")
# plt.legend(["B", "A"])
# plt.show()








#DELTA BATCH
plt.ion()

mse = []
accuracyA = []
accuracyB = []
for epoch in range(epochs):

    W += - eta * (W@X - T) @ X.T

    mse.append(np.mean((W @ X - T) ** 2))
    plt.clf()
    plot_data(X, T)
    plot_boundaries(W)
    plt.title(f"epoch:{epoch}")
    plt.pause(0.2)

    print(f"EPOCH={epoch}, accuracy:")
    accuracy = missclassified_rate(X, T, W)
    print(accuracy)
    accuracyA.append(accuracy[1, 1])
    accuracyB.append(accuracy[0, 0])


plt.ioff()

plt.figure()
plt.plot(range(epochs), mse)
plt.title("MSE")
plt.show()

plt.figure()
plt.plot(range(epochs), accuracyA)
plt.plot(range(epochs), accuracyB)
plt.title("Accuracy")
plt.legend(["B", "A"])
plt.show()