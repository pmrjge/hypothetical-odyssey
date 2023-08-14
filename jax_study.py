import jax.numpy as jnp
from jax import grad, jit, vmap, random, device_put, jacfwd, jacrev, lax
from timeit import default_timer as timer
import matplotlib.pyplot as plt

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


# Autovectorization with vmap()

mat = random.normal(key, (150, 100))
batched_x = random.normal(key, (10, 100))

def apply_matrix(v):
    return jnp.dot(mat, v)

def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
t0 = timer()
naively_batched_apply_matrix(batched_x).block_until_ready()
t1 = timer()
print(f'it took {t1-t0} ms.')

@jit
def batched_apply_matrix(v_batched):
    return jnp.dot(v_batched, mat.T)

print('Manually batched')
t0 = timer()
batched_apply_matrix(batched_x).block_until_ready()
t1 = timer()
print(f'it took {t1 - t0} ms.')

@jit
def vmap_batched_apply_matrix(v_batched):
    return vmap(apply_matrix)(v_batched)

print('Auto-vectorized with vmap')
t0 = timer()
vmap_batched_apply_matrix(batched_x).block_until_ready()
t1 = timer()
print(f'it took {t1 - t0} ms.')

# JAX vs Numpy
## JAX arrays are immutable
x_np = np.linspace(0, 10, 1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
plt.plot(x_np, y_np)

x_jnp = jnp.linspace(0, 10, 1000)
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)


## Numpy, lax & XLA

print(jnp.add(1, 1.0))
print(lax.add(jnp.float32(1), 1.0))

x = jnp.array([ 1, 2, 1])
y = jnp.ones(10)

print(jnp.convolve(x, y))

result = lax.conv_general_dilated(x.reshape(1, 1, 3).astype(float), y.reshape(1, 1, 10), window_strides=(1,), padding="same")

print(result[0, 0])

## JIT or not to JIT

def norm(X):
    X = X - X.mean(0)
    return X / X.std(0)

norm_compiled = jit(norm)

np.random.seed(1701)
X = jnp.array(np.random.rand(10000, 10))
print(np.allclose(norm(X), norm_compiled(X), atol=1E-6))

t0 = timer()
norm(X).block_until_ready()
t1 = timer()
print(f"Time using non jitted function: {t1 - t0} ms")

t0 = timer()
norm_compiled(X).block_until_ready()
t1 = timer()
print(f"Time using jitted function: {t1 - t0} ms")
