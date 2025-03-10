import numpy as np
import numpy.linalg as LA

def rayleigh(B, b) -> float:
    B = np.ndarray(B)
    b = np.ndarray(b)

    """Ser så att B är kvadratisk, att B och b har en delad dimension n eller m, och att b är normalvektor."""
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square.')
    if B.shape[0] != b.shape[0]:
        raise ValueError('B and b must share dimensions.')
    if np.linalg.norm(b) != 1.0:
        raise ValueError('b is not a normal vector (length of 1).')

    """b^T * B * b med matriser enligt numpydokumentation."""
    return float(b.T @ B @ b)



def max_eigenvalue(B: np.ndarray, p: int):
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square.')

    n = B.shape[0]
    b = np.random.randn(n, p) # random matrix with n rows, p columns
    b = LA.norm(b)

    prev_rayleigh = rayleigh(B, b)
    tolerance = 10**-p

    """Jämför med flera följande på varandra Rayleigh-kvoter för varje iteration av B."""
    while True:
        next_b = B @ b # [B][b]
        next_b /= LA.norm(next_b) # [b]/|[b]|

        curr_rayleigh = rayleigh(B, next_b)

        if abs(curr_rayleigh - prev_rayleigh) < tolerance:
            return curr_rayleigh, next_b

        prev_rayleigh = curr_rayleigh
        b = next_b


# (A)
B = np.array([[9, 5], [1, 5]])

