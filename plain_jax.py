import jax
from jax import numpy as jnp, grad, lax, jit, vmap, value_and_grad, jacfwd, jacrev, random as jr
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

# plt.scatter(xs, ys)
# plt.plot(xs, model(params, xs))
# plt.show()

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

#Automatic Vectorization with JAX
# Manual Vectorization

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

print(convolve(x, w))

xs = jnp.stack([x, x])
ws = jnp.stack([w, w])

def manually_batched_convolve(xs, ws):
    output = []
    for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
    return jnp.stack(output)

print(manually_batched_convolve(xs, ws))

def manually_vectorized_convolve(xs, ws):
    output = []
    for i in range(1, xs.shape[-1]-1):
        output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
    return jnp.stack(output, axis=1)

print(manually_vectorized_convolve(xs, ws))


# Automatic Vectorization

auto_batch_convolve = vmap(convolve)

print(auto_batch_convolve(xs, ws))

auto_batch_convolve_v2 = vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

print(auto_batch_convolve_v2(xst, wst))


batch_convolve_v3 = vmap(convolve, in_axes=[0, None])

print(batch_convolve_v3(xs, w))

# Combining transformations

jitted_batch_convolve = jit(auto_batch_convolve)

print(jitted_batch_convolve(xs, ws))

# Advanced Automatic Differentiation in JAX

# Higher-Order Derivatives

f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = grad(f)
d2fdx = grad(dfdx)
d3fdx = grad(d2fdx)
d4fdx = grad(d3fdx)

print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))

def hessian(f):
    return jacfwd(grad(f))

f = lambda x: jnp.dot(x, x)

print(hessian(f)(jnp.array([1., 2., 3.])))

# Higher Order Optimization

# lr = 0.01
# def meta_loss_fn(params, data):
#     grads = grad(loss_fn)(params, data)
#     return loss_fn(params - lr * grads, data)

# meta_grads = grad(meta_loss_fn)(params, data)

# Stopping gradients

value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])

s_tm1 = jnp.array([1. ,2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])

def td_loss(theta, s_tm1, r_t, s_t):
    v_tm1 = value_fn(theta, s_tm1)
    target = r_t + value_fn(theta, s_t)
    return (target - v_tm1) ** 2

td_update = grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

print(delta_theta)
#### jax.lax.stop_gradient is key to stop from flowing the gradient

def td_loss(theta, s_tm1, r_t, s_t):
    v_tm1 = value_fn(theta, s_tm1)
    target = r_t + value_fn(theta, s_t)
    return (lax.stop_gradient(target) - v_tm1) ** 2

td_update = grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

print(delta_theta)

# Straight-through estimator using stop_gradient

f = jnp.round

def straight_through_f(x):
    zero = x = lax.stop_gradient(x)
    return zero + lax.stop_gradient(f(x))

print("f(x)", f(3.2))
print("straight_through_f(x):", straight_through_f(3.2))

print("grad(f)(x): ", jax.grad(f)(3.2))
print("grad(straight_through_f)(x):", grad(straight_through_f)(3.2))


# Per Example gradients

perex_grads = jit(vmap(grad(td_loss), in_axes=(None, 0, 0, 0)))

batched_s_tm1 = jnp.stack([s_tm1, s_tm1])
batched_r_t = jnp.stack([r_t, r_t])
batched_s_t = jnp.stack([s_t, s_t])

print(perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t))


#Random Number in NumPy


np.random.seed(0)

def print_truncated_random_state():
    """To avoid spamming the outputs, print only the part of the state"""
    full_random_state = np.random.get_state()
    print(str(full_random_state[:460]), '...')

print_truncated_random_state()

np.random.seed(0)

print_truncated_random_state()

_ = np.random.uniform()

print_truncated_random_state()

np.random.seed(0)
print(np.random.uniform(size=3))

np.random.seed(0)
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))

np.random.seed(0)
print("all at once: ", np.random.uniform(size=3))


# Random numbers in JAX
import numpy as np
np.random.seed(0)
def bar(): return np.random.uniform()
def baz(): return np.random.uniform()

def foo(): return bar () + 2 * baz()

print(foo())
key =jr.PRNGKey(42)

print(key)

print(jr.normal(key))
print(jr.normal(key))

print("old key", key)

new_key, subkey = jr.split(key)
del key
normal_sample = jr.normal(subkey)
print(r"    \---SPLIT --> new key   ", new_key)
print(r"            \--> new subkey", subkey, "--> normal", normal_sample)

key = new_key

