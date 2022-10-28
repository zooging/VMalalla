import numpy as np
import numpy.random as rand

def decompose_to_LU(a):
    lu_matrix = np.matrix(np.zeros([a.shape[0], a.shape[1]]))
    n = a.shape[0]

    for k in range(n):
        # вычисдить все элементы к-ой строки
        for j in range(k, n):
            lu_matrix[k, j] = a[k, j] - lu_matrix[k, :k] * lu_matrix[:k, j]
        # вычислить все элементы к-го столбца
        for i in range(k + 1, n):
            lu_matrix[i, k] = (a[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

    return lu_matrix

def get_L(m):
    L = m.copy()
    for i in range(L.shape[0]):
            L[i, i] = 1
            L[i, i+1 :] = 0
    return np.matrix(L)


def get_U(m):
    U = m.copy()
    for i in range(1, U.shape[0]):
        U[i, :i] = 0
    return U

def solve_LU(lu_matrix, b):
    # get supporting vector y
    y = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(y.shape[0]):
        y[i, 0] = b[i, 0] - lu_matrix[i, :i] * y[:i]

    # get vector of answers x
    x = np.matrix(np.zeros([lu_matrix.shape[0], 1]))
    for i in range(1, x.shape[0] + 1):
        x[-i, 0] = (y[-i] - lu_matrix[-i, -i:] * x[-i:, 0] )/ lu_matrix[-i, -i]

    return x


print("Это а")
a = np.random.randint(1, 10, (4, 4))
##a = np.array([[48,2,3,1,3],[3,16,1,3,1],[1,3,16,3,2],[1,2,2,2,56]])
print(a)
print("Это б")
b = np.random.randint(1, 10, (4, 1))
print(b)

LU = decompose_to_LU(a)
L = get_L(LU)
print("Матрица L")
print(L)
U = get_U(LU)
print("Матрица U")
print(U)

print("Проверка,перемножили матрицы L,U")
print(np.dot(L, U))

print("Решение")
X = solve_LU(LU,b)
print(X)


##SS = [20]
##for i in range(1,4):
    ##for #j in range(1,4):
        ##SS[i] = SS[i] + (a[i,j]*X[j])
    ##(SS[i])






