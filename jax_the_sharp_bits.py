import numpy as np
from jax import grad, jit, lax, random as jr, numpy as jnp
import jax
from functools import partial


# Pure Functions

def impure_print_side_effect(x):
    print("Executing function")
    return x

print("First call: ", jit(impure_print_side_effect)(4.))
print("Second call: ", jit(impure_print_side_effect)(5.))
print("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))

g = 0.
def impure_uses_globals(x):
    return x + g

print("First call: ", jit(impure_uses_globals)(5.))

g = 10.

print("second call: ", jit(impure_uses_globals)(5.))

print("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))

g = 0.
def impuse_saves_global(x):
    global g
    g = x
    return x

print("First call: ", jit(impuse_saves_global)(4.))
print("Saved global: ", g)

def pure_uses_internal_state(x):
    state = dict(even=0, odd=0)
    for i in range(10):
        state['even' if i % 2 == 0 else 'odd'] += x
    return state['even'] + state['odd']

print(jit(pure_uses_internal_state)(5.))

array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i, x: x + array[i], 0))
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0))

def func11(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return carry + ae1 * ae2, carry
    return lax.scan(body, 0., (arr, ones))
# make_jaxpr(func11)(iter(range(16)), 5.)

array_operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
iter_operand = iter(range(10))
# lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand)


# In-Place Updates

np_array = np.zeros((3,3), dtype=np.float32)
print("original array")
print(np_array)

np_array[1, :] = 1.0
print("updated array:")
print(np_array)

jax_array = jnp.zeros((3,3), dtype=jnp.float32)

# jax_array[1, :] = 1.0

updated_array = jax_array.at[1, :].set(1.0)

print("updated array:\n", updated_array)
print("original array unchanged:\n", jax_array)

# Array updates with other operations
print("original array:")
jax_array = jnp.ones((5, 6))
print(jax_array)

new_jax_array = jax_array.at[::2, 3:].add(7.)
print("new array post-addition:")
print(new_jax_array)


# Out-of-bounds indexing

# np.arange(10)[11]
print(jnp.arange(10)[11])
print(jnp.arange(10.0).at[11].get())
print(jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan))


# Non-array inputs: NumPy vs JAX

print(np.sum([1, 2, 3]))

# jnp.sum([1, 2, 3])



# np Random Numbers


print(np.random.random())
print(np.random.random())
print(np.random.random())

# JAX PRNG

# It uses a modern threefry counter-based PRNG that's splittable

key = jr.PRNGKey(0)
key

print(jr.normal(key, shape=(1,)))
print(key)
print(jr.normal(key, shape=(1,)))
print(key)

print("old key", key)
k, sk = jr.split(key)
normal_pseudorandom = jr.normal(sk, shape=(1,))
print("SPLIT --> new key", k)
print("    new subkey", sk, "--> normal", normal_pseudorandom)

print("old key", k)
k, sk = jr.split(k)
normal_pseudorandom = jr.normal(sk, shape=(1,))
print("SPLIT --> new key", k)
print("    new subkey", sk, "--> normal", normal_pseudorandom)

key, *subkeys = jr.split(k, 4)
for subkey in subkeys:
    print(jr.normal(subkey, shape=(1,)))


# Structured control flow primitives

# cond

def cond(pred, true_fn, false_fn, operand):
    return true_fn(operand) if pred else false_fn(operand)

# Jax Equivalent
operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, operand)
lax.cond(False, lambda x: x+1, lambda x: x-1, operand)

# while_loop

def while_loop(cond_fn, body_fn, init_val):
    val = init_val
    while cond_fn(val):
        val = body_fn(val)
    return val

init_val = 0
cond_fn = lambda x: x<10
body_fn = lambda x: x+1
print(lax.while_loop(cond_fn, body_fn, init_val))

# fori_loop
def fori_loop(start, stop, body_fn, init_val):
    val = init_val
    for i in range(start, stop):
        val = body_fn(i, val)
    return val

init_val = 0
start = 0
stop = 10
body_fn = lambda i,x: x + i
print(lax.fori_loop(start, stop, body_fn, init_val))


# Dynamic Shapes

def nansum(x):
    mask = ~jnp.isnan(x)
    x_without_nans = x[mask]
    return x_without_nans.sum()

x = jnp.array([1,2,jnp.nan, 3, 4])
print(nansum(x))

# This will error: jax.jit(nansum)(x)

@jax.jit
def nansum_2(x):
    mask = ~jnp.isnan(x)
    return jnp.where(mask, x, 0.).sum()

print(nansum_2(x))

# NaNs

# JAX_DEBUG_NANS=True in our environment variable

# from jax import config
# config.update("jax_debug_nans", True)

# from jax import config
# config.parse_flags_with_absl()
#add option --jax_debug_nans=True

# Double precision (64 bit floats or integers)

# JAX_ENABLE_X64=True at the environment variable

#Manually
# from jax import config
# config.update("jax_enable_x64", True)

# from jax import config
# config.config_with_absl()

# from jax import config
# if __name__ == '__main__':
#   config.parse_flags_with_absl():


# Caveats
