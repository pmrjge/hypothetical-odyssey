import jax
from jax import numpy as jnp, random as jr, nn as jnn, lax, jit, grad, vmap, jacfwd, jacrev, pmap, value_and_grad
import numpy as np

x = np.arange(5)
w = np.array([2., 3., 4.])

def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

print(convolve(x, w))

n_devices = jax.local_device_count()
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)

print(xs)

print(ws)

print(vmap(convolve)(xs,ws))

print(pmap(convolve)(xs, ws))

print(pmap(convolve)(xs, pmap(convolve)(xs, ws)))

print(pmap(convolve, in_axes=(0, None)(xs, w)))

# pmap ==> ~jit

# Communication between devices

def normalized_convolution(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    output = jnp.array(output)
    return output / lax.psum(output, axis_name='p')

print(pmap(normalized_convolution, axis_name='p')(xs, ws))

# Nesting pmap and vmap

# vmap(pmap(f, axis_name='i'), axis_name='j')

