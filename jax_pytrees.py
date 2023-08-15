import jax
from jax import numpy as jnp, random as jr, lax, jacfwd, jacrev, jit, grad, vmap, pmap, value_and_grad, nn as jnn
from timeit import timeit
import matplotlib.pyplot as plt
import numpy as np
import collections
from dataclasses import dataclass
from typing import Iterable


dtimer = lambda test: timeit(lambda: test(), number=100)


# What is a PyTree?

example_trees = [
    [1, 'a', object()],
    (1, (2, 3), ()),
    [1, {'k1': 2, 'k2': (3, 4)}, 5],
    {'a': 2, 'b': (2, 3)},
    jnp.array([1, 2, 3]),
]

for pytree in example_trees:
  leaves = jax.tree_util.tree_leaves(pytree)
  print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

  list_of_lists = [
    [1, 2, 3],
    [1, 2],
    [1, 2, 3, 4]
]

print(jax.tree_map(lambda x: x*2, list_of_lists))

another_list_of_lists = list_of_lists
print(jax.tree_map(lambda x,y: x+y, list_of_lists, another_list_of_lists))

# Example: ML model parameters

def init_mlp_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in), biases=np.ones(shape=(n_out,))))
    return params

params = init_mlp_params([1, 128, 128, 1])

print(jax.tree_map(lambda x: x.shape, params))

def forward(params, x):
    *hidden, last = params
    for layer in hidden:
        x = jnn.relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

LEARNING_RATE = 0.0001

@jit
def update(params, x, y):
    grads = grad(loss_fn)(params, x, y)
    return jax.tree_map(lambda p, g: p - LEARNING_RATE * g, params, grads)


xs = np.random.normal(size=(128, 1))
ys = xs ** 2

for _ in range(1000):
    params = update(params, xs, ys)

plt.scatter(xs, ys)
plt.scatter(xs, forward(params, xs), label='Model prediction')
plt.legend()
plt.show()

# Key paths

ATuple = collections.namedtuple("ATuple", ('name'))

tree = [1, {'k1': 2, 'k2': (3,4)}, ATuple('foo')]
flattened, _ = jax.tree_util. tree_flatten_with_path(tree)
for key_path, value in flattened:
    print(f'Value of tree {jax.tree_util.keystr(key_path)}: {value}')


# custom pytree nodes

@dataclass
class MyContainer:
    name: str
    a: int
    b: int
    c: int


pytree = [MyContainer(name='Alice', a=1, b=2, c=3), MyContainer(name='Bob', a=4, b=5, c=6)]
print(jax.tree_util.tree_leaves(pytree))


try:
    jax.tree_map(lambda x: x+1, pytree)
except TypeError as e:
    print(f'TypeError: {e}')

def flatten_MyContainer(container) -> tuple[Iterable[int], str]:
    flat_contents = [container.a, container.b, container.c]

    aux_data = container.name
    return flat_contents, aux_data

def unflatten_MyContainer(aux_data: str, flat_contents: Iterable[int])-> MyContainer:
    return MyContainer(aux_data, *flat_contents)

jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer)

print(jax.tree_util.tree_leaves(pytree))

class MyKeyPathContainer(MyContainer):
    pass

def flatten_with_keys_MyKeyPathContainer(container) -> tuple[Iterable[int], str]:
    flat_contents = [(jax.tree_util.GetAttrKey('a'), container.a), (jax.tree_util.GetAttrKey('b'), container.b), (jax.tree_util.GetAttrKey('c'), container.c)]
    aux_data = container.name
    return flat_contents, aux_data

def unflatten_MyKeyPathContainer(aux_data: str, flat_contents: Iterable[int]) -> MyKeyPathContainer:
    return MyKeyPathContainer(aux_data, *flat_contents)

jax.tree_util.register_pytree_with_keys(MyKeyPathContainer, flatten_with_keys_MyKeyPathContainer, unflatten_MyKeyPathContainer)

print(jax.tree_util.tree_leaves([MyKeyPathContainer('Alice', 1, 2, 3), MyKeyPathContainer('Bob', 4, 5, 6)]))

# Common pytree gotchas and patterns

# Gotchas

# Mistaking nodes for leaves

a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

shapes = jax.tree_map(lambda x: x.shape, a_tree)
print(jax.tree_map(jnp.ones, shapes))

# Handling of None

print(jax.tree_util.tree_leaves([None, None, None]))

# Patterns

# Transposing trees

def tree_transpose(list_of_trees):
    return jax.tree_map(lambda *xs: list(xs), *list_of_trees)

episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
print(tree_transpose(episode_steps))

print(jax.tree_transpose(
    outer_treedef = jax.tree_structure([0 for e in episode_steps]),
    inner_treedef = jax.tree_structure(episode_steps[0]),
    pytree_to_transpose = episode_steps
))

