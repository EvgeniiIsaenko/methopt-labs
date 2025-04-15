import numpy as np
import matplotlib.pyplot as plt

def generate_positive_definite_matrix(size=6):
    A = np.random.rand(size, size)
    
    # симметричная, положительна определенная (A * A^T)
    A = np.dot(A, A.T)
    
    return A

def gradient(x, A, b):
    # градиент функции f(x)
    return 0.5 * (A.T + A) @ x + b 

def function_value(x, A, b):
    # значения функции f(x)
    return 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(b, x)

def gradient_descent(A, b, x0, learning_rate=1e-4, tolerance=1e-5, max_iterations=1000000):
    # метод градиентного спуска для минимизации функции f(x)
    x = x0
    f_values = [function_value(x, A, b)]  # Список для хранения значений функции
    x_values = [x]
    for _ in range(max_iterations):
        grad = gradient(x, A, b)
        x_new = x - learning_rate * grad
        f_values.append(function_value(x_new, A, b))
        x_values.append(x_new)
        
        # проверка на сходимость
        if np.linalg.norm(x_new - x) < tolerance:
            break
        
        x = x_new
    return x, f_values, x_values, len(f_values)

A = generate_positive_definite_matrix()
b = np.array([0.53453240, 0.60307323, 0.10238401, 0.63493003, 0.24934701, 0.10047803])
x0 = np.array([0.01247024, 0.10472905, 0.00346109, 0.57095621, 0.26825007, 0.00579221])

print('исходная матрица', np.around(A, 3))
print('b: ', np.around(b, 3))
print('x0: ', np.around(x0, 3))

x_exact = -2 * np.linalg.inv((A.T + A)) @ b
solution, function_values, arg_values, N = gradient_descent(A, b, x0)

print("x N:", solution)
print("(x*):", x_exact)
print("значение функционала в x*", function_value(x_exact, A, b))
print("Погрешность:", solution - x_exact)
f_dif = function_value(solution, A, b) - function_value(x_exact, A, b)
print("f_dif: ", f_dif)

# Вывод значений на определенных шагах
indices = [N // 4, N // 2, 3 * N // 4, N - 1]
for idx in indices:
    if idx < len(function_values):
        print(f"x({idx+1}): {arg_values[idx]}")
        print(f"f(x({idx+1})): {function_values[idx]}")

# Построение графика
plt.plot(range(len(function_values)), function_values)
plt.xlabel('Номер шага')
plt.ylabel('Значение функции f(x)')
plt.title('Зависимость значения функции от номера шага')
plt.grid()
plt.show()

