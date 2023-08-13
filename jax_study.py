import jax.numpy as jnp
from jax import grad, jit, vmap, random, device_put
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

