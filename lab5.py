import numpy.linalg as LA
import numpy as np

def rayleigh(B, b):
    return b.T @ B @ b # @ is matrix multiplication

def power_iteration(B, p):
    # Initialize "guess" eigenvalue and eigenvector
    b = np.ones((B.shape[1], 1))
    eigenval = 10
    while True:
        # Power iteration formula
        b_new = (B @ b) / (LA.norm(B @ b))
        
        # Check if close to solution
        if abs(rayleigh(B, b) - rayleigh(B, b_new)) < 10**-p:
            break
        
        # Update eigenvalue
        eigenval = rayleigh(B, b_new)
        b = b_new

    return b_new, eigenval


# a)
print("a)")
B = np.array([[9, 5], 
              [1, 5]])
eigenvector, eigenval = power_iteration(B, 6)
print("The largest eigenvalue is", round(eigenval[0, 0], 2), "using power iteration (potensmetoden).")
print("The largest eigenvalue is", max(LA.eig(B)[0]), "using LA.eig.")
print()

# b)
print("b)")
A = np.random.rand(500, 500) # Initialize random matrix of size 500x500
B = A + A.T
v,  λ = power_iteration(B, 6)
magnitude = LA.norm(B @ v -  λ * v) # Magnitude should be close to zero if correct solution
print("The largest eigenvalue for B is", round( λ[0, 0], 2))
# print("The eigenvector is", v) # (this prints 500 rows)
print("Bv - λv gives us", magnitude)
