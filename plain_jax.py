import jax
from jax import numpy as jnp, grad, lax, jit, vmap
from timeit import timeit

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