from typing import Any
import tensorflow_datasets as tfds
import tensorflow as tf

import jax
import flax
from jax import numpy as jnp, random as jr
from flax import linen as nn, struct
from flax.training import train_state
import optax

from clu import metrics

import matplotlib.pyplot as plt


def get_datasets(num_epochs, batch_size):
    """Load MNIST train and test datasets into memory."""

    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')

    train_ds = train_ds.map(lambda sample: {'image': tf.cast(sample['image'], tf.float32), 'label': sample['label']})
    test_ds = test_ds.map(lambda sample: {'image': tf.cast(sample['image'], tf.float32), 'label': sample['label']})

    train_ds = train_ds.repeat(num_epochs).shuffle(1024).batch(batch_size, drop_reminder=True).prefetch(1)
    test_ds = test_ds.shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, test_ds

class CNN(nn.Module):
    """A Simple CNN Model."""

    @nn.compact
    def __call__(self,x):
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2,2))
        x = x.reshape((x.shape[0], -1)) # flatten to 2D Tensor

        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x
    
# Define model

cnn = CNN()
print(cnn.tabulate(jr.PRNGKey(0), jnp.ones((1, 28, 28, 1))))

# Create TrainState

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones((1, 28, 28, 1)))['params'] # Initializes the params for the model
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=module.apply, params=params, metrics=Metrics.empty())

# Training Step

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
        return loss
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


# Metric Computation

@jax.jit
def compute_metrics(*, state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()

    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=batch['label'], loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state

# Download data

num_epochs = 10
batch_size = 32

train_ds, test_ds = get_datasets(num_epochs, batch_size)

# Seed Randomness

tf.random.set_seed(0)
init_rng = jr.PRNGKey(0)

# Initialize the TrainState

learning_rate = 0.01
momentum = 0.9
state = create_train_state(cnn, init_rng, learning_rate, momentum)
del init_rng

# 10. Train and Evaluate

num_steps_per_epoch = train_ds.cardinality().numpy() // num_epochs

metrics_history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [], 'test_accuracy': []}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    state = train_step(state, batch)
    state = compute_metrics(state=state, batch=batch)

    # When computing over the whole batch, compute metrics for train and testing
    if(step + 1) % num_steps_per_epoch == 0:
        for metric, value in state.metrics.compute().item():
            metrics_history[f'train_{metric}'].append(value)
        state = state.replace(metrics=state.metrics.empty())

        test_state = state
        for test_batch in test_ds.as_numpy_iterator():
            test_state = compute_metrics(state=test_state, batch=test_batch)

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)

        print(f"train epoch: {(step+1) // num_steps_per_epoch}, ", f"loss: {metrics_history['train_loss'][-1] * 100.0}")
        print(f"test epoch: {(step + 1) // num_steps_per_epoch}, ", f"loss: {metrics_history['test_loss'][-1]}, ", f"accuracy: {metrics_history['test_accuracy'][-1] * 100.0}")


