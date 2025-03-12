import numpy as np
import numpy.linalg as LA

def rayleigh(B, b):
    # Ensure B is square, B and b have matching dimensions, and b is a unit vector.
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square.')
    if B.shape[0] != b.shape[0]:
        raise ValueError('B and b must share dimensions.')
    if not np.isclose(LA.norm(b), 1.0):
        raise ValueError('b is not a normal vector (length of 1).')

    return b.T @ B @ b # Computes Rayleigh quotient [B]^T[b][B].
                       # @ is matrix multiplication. '.T' is transposition of a matrix.


def max_eigenvalue(B, p):
    if B.shape[0] != B.shape[1]:
        raise ValueError('B is not square.')

    b = np.ones((B.shape[0], 1))  # Initialize b as column vector of ones as a starting vector.
    b /= LA.norm(b)  # Normalize b at initialization.
    prev_rayleigh = rayleigh(B, b) # Initial Rayleigh quotient.
    tolerance = 10 ** -p

    while True:
        # Step-by-step power iteration
        next_b = B @ b # [b] = [B][b]
        # Normalize next_b as a unit vector.
        next_b /= LA.norm(next_b)  # [b] = [b]/|[b]|

        curr_rayleigh = rayleigh(B, next_b) # Updated iteration of Rayleigh quotient.

        if abs(curr_rayleigh - prev_rayleigh) < tolerance:
            return next_b, curr_rayleigh

        prev_rayleigh = curr_rayleigh
        b = next_b

# a)
print("a)")
B = np.array([[9, 5],
              [1, 5]])

eigenvector, eigenval = max_eigenvalue(B, 6)
print("Largest eigenvalue through power iteration: ", round(eigenval[0,0], 2))
print("The largest eigenvalue through LA.eig: ", max(LA.eig(B)[0]))

# b)
print("\nb)")
A = np.random.rand(500, 500)
B = A + A.T  # Make B symmetric

v, eigenvalue = max_eigenvalue(B, 6)
magnitude = LA.norm(B @ v - eigenvalue * v)  # Should be close to zero

print("Largest eigenvalue for B:", round(eigenvalue[0,0], 2))
print("Bv - λv = ", magnitude)
