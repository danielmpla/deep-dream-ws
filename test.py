from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from entity import mongo_setup
from repository.queue_repository import QueueRepository
from io import BytesIO

import numpy as np
import time
import matplotlib as mpl

import PIL.Image

from tensorflow.keras.preprocessing import image

OCTAVE_SCALE = 1.30

class TiledGradients(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),)
    )
    def __call__(self, img, tile_size=512):
        if tile_size > img.shape[0]:
            tile_size = 128

        shift_down, shift_right, img_rolled = DeepDreamWS.random_roll(img, tile_size)

        # Initialize the image gradients to zero.
        gradients = tf.zeros_like(img_rolled)

        # Skip the last tile, unless there's only one tile.
        xs = tf.range(0, img_rolled.shape[0], tile_size)[:-1]
        if not tf.cast(len(xs), bool):
            xs = tf.constant([0])
        ys = tf.range(0, img_rolled.shape[1], tile_size)[:-1]
        if not tf.cast(len(ys), bool):
            ys = tf.constant([0])

        for x in xs:
            for y in ys:
                # Calculate the gradients for this tile.
                with tf.GradientTape() as tape:
                    # This needs gradients relative to `img_rolled`.
                    # `GradientTape` only watches `tf.Variable`s by default.
                    tape.watch(img_rolled)

                    # Extract a tile out of the image.
                    img_tile = img_rolled[x:x + tile_size, y:y + tile_size]
                    loss = DeepDreamWS.calc_loss(img_tile, self.model)

                # Update the image gradients for this tile.
                gradients = gradients + tape.gradient(loss, img_rolled)

        # Undo the random shift applied to the image and its gradients.
        gradients = tf.roll(tf.roll(gradients, -shift_right, axis=1), -shift_down, axis=0)

        # Normalize the gradients.
        gradients /= tf.math.reduce_std(gradients) + 1e-8

        return gradients


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
                tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int32),
                tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):
        print("Tracing")
        loss = tf.constant(0.0)
        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                # This needs gradients relative to `img`
                # `GradientTape` only watches `tf.Variable`s by default
                tape.watch(img)
                loss = DeepDreamWS.calc_loss(img, self.model)

            # Calculate the gradient of the loss with respect to the pixels of the input image.
            gradients = tape.gradient(loss, img)

            # Normalize the gradients.
            gradients /= tf.math.reduce_std(gradients) + 1e-8

            # In gradient ascent, the "loss" is maximized so that the input image increasingly "excites" the layers.
            # You can update the image by directly adding the gradients (because they're the same shape!)
            img = img + gradients * step_size
            img = tf.clip_by_value(img, -1, 1)

        return loss, img


class DeepDreamWS:

    def __init__(self):
        base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

        # Maximize the activations of these layers
        names = ['mixed3', 'mixed5']
        layers = [base_model.get_layer(name).output for name in names]

        # Create the feature extraction model
        dream_model = tf.keras.Model(inputs=base_model.input, outputs=layers)
        self.deepdream = DeepDream(dream_model)

        self.get_tiled_gradients = TiledGradients(dream_model)

    def open_image(self, image_id, max_dim=None):
        job = QueueRepository.get_job_by_id(image_id)

        img_bytes = BytesIO(job.base_image)

        img = PIL.Image.open(img_bytes)
        if max_dim:
            img.thumbnail((max_dim, max_dim))
        return np.array(img)

    # Normalize an image
    def deprocess(self, img):
        img = 255 * (img + 1.0) / 2.0
        return tf.cast(img, tf.uint8)

    # Display an image
    def show_and_store(self, img, id):
        processed_img = PIL.Image.fromarray(np.array(img))

        job = QueueRepository.get_job_by_id(id)

        processed_img_bytes = BytesIO()
        processed_img.save(processed_img_bytes, 'jpeg')

        job.finished = True
        job.computed_image = processed_img_bytes.getvalue()

        job.save()

        processed_img_bytes.close()

    def set_canceled(self, id):
        job = QueueRepository.get_job_by_id(id)

        job.canceled = True

        job.save()

    @classmethod
    def calc_loss(cls, img, model):
        # Pass forward the image through the model to retrieve the activations.
        # Converts the image into a batch of size 1.
        img_batch = tf.expand_dims(img, axis=0)
        layer_activations = model(img_batch)

        losses = []
        for act in layer_activations:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)

        return tf.reduce_sum(losses)

    def run_deep_dream_simple(self, img, steps=100, step_size=0.01):
        # Convert from uint8 to the range expected by the model.
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        img = tf.convert_to_tensor(img)
        step_size = tf.convert_to_tensor(step_size)
        steps_remaining = steps
        step = 0
        while steps_remaining:
            if steps_remaining > 100:
                run_steps = tf.constant(100)
            else:
                run_steps = tf.constant(steps_remaining)
            steps_remaining -= run_steps
            step += run_steps

            loss, img = self.deepdream(img, run_steps, tf.constant(step_size))

            print("Step {}, loss {}".format(step, loss))

        result = self.deprocess(img)

        return result

    def run_deep_dream_with_octaves(self, img, steps_per_octave=100, step_size=0.01,
                                    octaves=range(-3, 3), octave_scale=1.3):
        base_shape = tf.shape(img)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        initial_shape = img.shape[:-1]
        img = tf.image.resize(img, initial_shape)
        for octave in octaves:
            # Scale the image based on the octave
            new_size = tf.cast(tf.convert_to_tensor(base_shape[:-1]), tf.float32) * (octave_scale ** octave)
            img = tf.image.resize(img, tf.cast(new_size, tf.int32))

            for step in range(steps_per_octave):
                gradients = self.get_tiled_gradients(img)
                img = img + gradients * step_size
                img = tf.clip_by_value(img, -1, 1)

        result = self.deprocess(img)
        return result

    @classmethod
    def random_roll(cls, img, maxroll):
        # Randomly shift the image to avoid tiled boundaries.
        shift = tf.random.uniform(shape=[2], minval=-maxroll, maxval=maxroll, dtype=tf.int32)
        shift_down, shift_right = shift[0], shift[1]
        img_rolled = tf.roll(tf.roll(img, shift_right, axis=1), shift_down, axis=0)
        return shift_down, shift_right, img_rolled

    def run(self, id: str):
        try:
            original_img = self.open_image(id)

            img = tf.constant(np.array(original_img))
            base_shape = tf.shape(img)[:-1]

            # shift_down, shift_right, img_rolled = self.random_roll(np.array(original_img), 128)

            start = time.time()

            img = self.run_deep_dream_with_octaves(img=original_img, step_size=0.01)

            end = time.time()

            print(end - start)

            img = tf.image.resize(img, base_shape)
            img = tf.image.convert_image_dtype(img / 255.0, dtype=tf.uint8)
            self.show_and_store(img, id)
        except Exception as e:
            self.set_canceled(id)
