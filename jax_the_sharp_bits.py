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