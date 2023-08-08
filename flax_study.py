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

from einops import rearrange, repeat, reduce


def get_datasets(num_epochs, batch_size):
    """Load MNIST train and test datasets into memory."""

    train_ds = tfds.load('mnist', split='train')
    test_ds = tfds.load('mnist', split='test')

    train_ds = train_ds.map(lambda sample: {'image': tf.cast(sample['image'], tf.float32), 'label': sample['label']})
    test_ds = test_ds.map(lambda sample: {'image': tf.cast(sample['image'], tf.float32), 'label': sample['label']})

    train_ds = train_ds.repeat(num_epochs).shuffle(1024).batch(batch_size, drop_remainder=True).prefetch(1)
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
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty())

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

def training_fn(state, train_ds, test_ds, metrics_history, num_steps_per_epoch):
    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        state = train_step(state, batch)
        state = compute_metrics(state=state, batch=batch)

        # When computing over the whole batch, compute metrics for train and testing
        if(step + 1) % num_steps_per_epoch == 0:
            for metric, value in state.metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value)
            state = state.replace(metrics=state.metrics.empty())

            test_state = state
            for test_batch in test_ds.as_numpy_iterator():
                test_state = compute_metrics(state=test_state, batch=test_batch)

            for metric, value in test_state.metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)

            print(f"train epoch: {(step+1) // num_steps_per_epoch}, ", f"loss: {metrics_history['train_loss'][-1] * 100.0}")
            print(f"test epoch: {(step + 1) // num_steps_per_epoch}, ", f"loss: {metrics_history['test_loss'][-1]}, ", f"accuracy: {metrics_history['test_accuracy'][-1] * 100.0}")

    return state, metrics_history

state, metrics_history = training_fn(state, train_ds, test_ds, metrics_history, num_steps_per_epoch)

# 11. Visualize metrics

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.set_title('Loss')
ax2.set_title('Accuracy')
for dataset in ('train', 'test'):
    ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
    ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')
ax1.legend()
ax2.legend()
plt.show()
plt.clf()

@jax.jit
def pred_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return logits.argmax(axis=1)

test_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(state, test_batch)

fig, axs = plt.subplots(5, 5, figsize=(12, 12))
for i, ax in enumerate(axs.flatten()):
    ax.imshow(test_batch['image'][i, ..., 0], cmap='gray')
    ax.set_title(f"label={pred[i]}")
    ax.axis('off')

def patch_div(x: jnp.array, num_patches):
    img_size = x.shape[1]
    patch_size = img_size // num_patches
    b, h, w, c = x.shape
    x = x.reshape((b, patch_size, num_patches, patch_size, num_patches, c))
    x = x.transpose((0, 1, 3, 2, 4, 5))
    return x


class AttentionBlock(nn.Module):
    embed_dim: int
    hidden_dim: int
    num_heads: int
    dropout_prob: float = 0.0
    train: bool = True

    @nn.compact
    def __call__(self, x):
        x0 = nn.LayerNorm()(x)
        x_attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(inputs_q=x0, inputs_kv=x0)
        x = x + nn.Dropout(self.dropout_prob)(x_attn, determinitic=not self.train)

        x_out = x
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout_prob)(x, deterministic=not self.train)
        x = nn.Dense(self.embed_dim)(x)
        x = x + nn.Dropout(self.dropout_prob)(x_out, deterministic=not self.train)
        return x
    
class VisionTransformer(nn.Module):
    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_channels: int = 1
    num_layers: int
    num_classes: int
    patch_size: int
    num_patches: int
    dropout_prob: float = 0.0
    train: bool = True

    @nn.compact
    def __call__(self, x):

        x = patch_div(x, self.num_patches)
        b, t = x.shape[:2]
        x = nn.Dense(self.embed_dim)(x)

        cls_token = self.param('cls_token', nn.initializers.normal(stddev=1.0), shape=(1, 1, self.embed_dim)).repeat(b, axis=0)
        x = jnp.concatenate((cls_token, x), axis=1)
        x = x + self.param('pos_embedding', nn.initializers.normal(stddev=1.0), shape=(1, 1+self.num_patches, self.embed_dim))[:, t+1]

        x = nn.Dropout(self.dropout_prob)(x, deterministic=not self.train)
        for _ in range(self.num_layers):
            x = AttentionBlock(embed_dim=self.embed_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, dropout_prob=self.dropout_prob, train=self.train)(x)
        
        cls = x[:,0]
        out = nn.LayerNorm()(cls)
        out = nn.Dense(self.num_classes)()

        return out


