import numpy as np
from scipy.optimize import minimize


A = np.array([
    [4, 1, 2, 3], 
    [1, 3, 4, 5], 
    [2, 4, 6, 7], 
    [3, 5, 7, 8]
]) 
b = np.array([1, 2, 3, 4])  
x0 = np.array([0, 0, 0, 0]) 
r = 1  


def f(x):
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)

def system_equations(x, y):
    eq1 = np.dot(A + 2 * y * np.eye(len(x)), x) + b + 2 * y * x0
    eq2 = np.linalg.norm(x - x0)**2 - r**2
    return np.concatenate((eq1, [eq2]))

def jacobian(x, y):
    J = np.zeros((len(x) + 1, len(x) + 1))
    J[:len(x), :len(x)] = A + 2 * y * np.eye(len(x))
    J[:len(x), len(x)] = 2 * (x - x0)
    J[len(x), :len(x)] = 2 * (x - x0).T
    return J

def newton_method(x0, y0, tol=1e-6, max_iter=100):
    x = np.concatenate((x0, [y0]))
    for i in range(max_iter):
        f_x = system_equations(x[:-1], x[-1])
        J_x = jacobian(x[:-1], x[-1])
        delta_x = -np.linalg.solve(J_x, f_x)
        x_new = x + delta_x
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = x_new
    return x


initial_guesses = [
    np.array([1, 0, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, 0, 1, 0]),
    np.array([0, 0, 0, 1]),
    np.array([-1, 0, 0, 0]),
    np.array([0, -1, 0, 0]),
    np.array([0, 0, -1, 0]),
    np.array([0, 0, 0, -1])
]

results = []

x_star = -np.linalg.solve(A^(-1), b)
print(f'x_star: {x_star}')
if np.linalg.norm(x_star - x0) <= r:
    results.append((None, None, x_star, 0, f(x_star)))

for i, x0_guess in enumerate(initial_guesses):
    y_start = np.random.uniform(0.01, 10)
    print(f'y_start: {y_start}')
    x_opt = newton_method(x0_guess, np.random.uniform(0.01, 10))
    x_opt_x = x_opt[:-1]
    x_opt_y = x_opt[-1]
    results.append((i, x0_guess, x_opt_x, x_opt_y, f(x_opt_x)))

print("points suspected of being optimum:")
print("i | initial guess | x_opt | y_opt | f(x_opt) ")
print("------------------------------------------------------------------------")
for i, x0_guess, x_opt, y_opt, f_x_opt in results:
    print(f"{i} | {x0_guess} | {x_opt} | {y_opt} | {f_x_opt}")

min_value = float('inf')
min_point = None
min_y = None
for i, x0_guess, x_opt, y_opt, f_x_opt in results:
    if np.linalg.norm(x_opt - x0) <= r and f_x_opt < min_value:
        min_value = f_x_opt
        min_point = x_opt
        min_y = y_opt

print(f"\nf_min: {min_value}")
print(f"x_min: {min_point}")
print(f"y_min: {min_y}")

# comparison with the scipy result  
def scipy_minimize():
    constraints = {'type': 'eq', 'fun': lambda x: np.linalg.norm(x - x0) - r}
    res = minimize(f, x0, method='SLSQP', constraints=constraints)
    return res.fun, res.x

scipy_min_value, scipy_min_point = scipy_minimize()

print(f"f(x_min_scipy): {scipy_min_value}")
print(f"x_min_scipy: {scipy_min_point}")

print(f"|f(x_min) - f(x_min_scipy)|: {abs(min_value - scipy_min_value)}")
print(f"norm(x_min - x_min_scipy): {np.linalg.norm(min_point - scipy_min_point)}")
