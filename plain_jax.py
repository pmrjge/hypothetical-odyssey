import jax
from jax import numpy as jnp, grad, lax, jit, vmap, value_and_grad
from timeit import timeit
import matplotlib.pyplot as plt
import numpy as np

dtimer = lambda test: timeit(lambda: test(), number=100)

x = jnp.arange(10)
print(x)

long_vector = jnp.arange(int(1e7))
print(f"{dtimer(lambda: jnp.dot(long_vector, long_vector).block_until_ready())} ms")

# JAX transformations: grad

def sum_of_squares(x):
    return jnp.sum(x**2)

d_sum_of_squares = grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])

print(sum_of_squares(x))
print(d_sum_of_squares(x))

def sum_squared_error(x, y):
    return jnp.sum((x-y)**2.0)

d_sum_squared_error = grad(sum_squared_error)

y = jnp.asarray([1.1, 2.1, 3.1, 4.1])

print(d_sum_squared_error(x, y))

partialdx_sum_squared_error = grad(sum_squared_error, argnums=(0, 1))
print(partialdx_sum_squared_error(x, y))

# Value and Grad

v, g = value_and_grad(sum_squared_error)(x, y)
print(v, g)


# Auxiliary Data

def squared_error_with_aux(x, y):
    return sum_squared_error(x, y), x - y

val = grad(squared_error_with_aux, has_aux=True)(x, y)
print(val)

# Differences from NumPy

# My First JAX training loop

xs = np.random.normal(size=(100,))
noise = np.random.normal(scale=0.1, size=(100,))
ys = xs * 3 - 1 + noise

#plt.scatter(xs, ys)

def model(params, x):
    w, b = params
    return w * x + b

def loss_fn(params, x, y):
    prediction = model(params, x)
    return jnp.mean((prediction - y) ** 2)

def update(params, x, y, lr=0.01):
    return params - lr * grad(loss_fn)(params, x, y)

params = jnp.array([1. ,1.])
for _ in range(1000):
    params = update(params, xs, ys)

plt.scatter(xs, ys)
plt.plot(xs, model(params, xs))
plt.show()

w, b = params
print(f"w: {w:<.2f}, b: {b:<.2f}")

#Just In Time Compilation with JAX
# How JAX transforms work

def selu(x, alpha=1.67, lambda_=1.05):
    return lambda_ * jnp.where(x>0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
print(f"Time without JIT: {dtimer(lambda: selu(x).block_until_ready()):<.3f} ms on average")

jselu = jit(selu)

# Compiling code for first time (optimizing python and jax operations)
jselu(x).block_until_ready()

print(f"Time with JIT: {dtimer(lambda: jselu(x).block_until_ready()):<.3f} ms on average")