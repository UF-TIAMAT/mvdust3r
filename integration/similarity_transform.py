import numpy as np

# Points needs to be non-colinear for Kabsch algorithm to work.

def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t


if __name__ == "__main__":

    # R, S, T 

    source = np.array([[1, 2, 3], [2, 3, 4], [0, 0, 1]])
    target = np.array([[2, 3, 4], [3, 4, 5], [1, 1, 2]])

    R, s, t = kabsch_umeyama(source, target)


    B = np.array([ np.round(t + s * R @ b, 8) for b in target])
    print("Source:\n", source)
    print("Target:\n", target)
    print("Transformed Source:\n", B)

    print("Rotation:\n", R)
    print("Scale:", s)
    print("Translation:\n", t)