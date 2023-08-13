import jax.numpy as jnp
from jax import grad, jit, vmap, random, device_put, jacfwd, jacrev
from timeit import default_timer as timer

# Generate random data in the following examples

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

size = 3000
x = random.normal(key, (size, size), dtype=jnp.float32)
ti = timer()
jnp.dot(x, x.T).block_until_ready()
tf = timer()
print(f"it took {tf - ti} ms using JAX")

import numpy as np
x = np.random.normal(size=(size, size)).astype(np.float32)
ti = timer()
jnp.dot(x, x.T).block_until_ready()
tf = timer()
print(f"it took {tf - ti} ms using numpy backend")

x = np.random.normal(size=(size, size)).astype(np.float32)
x = device_put(x)
ti = timer()
jnp.dot(x, x.T).block_until_ready()
tf = timer()
print(f"it took {tf - ti} ms using numpy backend with offloading to gpu")

# Using jit() to speed up functions

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = random.normal(key, (1000000,))
ti = timer()
selu(x).block_until_ready()
tf = timer()
print(f"it took {tf - ti} ms to execute code in jax without jit")

selu_jit = jit(selu)
ti = timer()
selu_jit(x).block_until_ready()
tf = timer()
print(f"it took {tf - ti} ms to execute code in jax with jit")


# Taking derivatives with grad()

def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))

def first_finite_differences(f, x):
    eps = 1e-3
    return jnp.array([(f(x + eps * v) - f(x - eps * v)) / (2 * eps) for v in jnp.eye(len(x))])

print(first_finite_differences(sum_logistic, x_small))

print(grad(jit(grad(jit(grad(sum_logistic)))))(1.0))

def hessian(f):
    return jit(jacfwd(jacrev(f)))

