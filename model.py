import tensorflow as tf
import numpy as np
import os
import time


def model_output(num:int):
    one_step_reloaded = tf.saved_model.load('one_step')
    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]
    for n in range(num):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)

    return str(tf.strings.join(result)[0].numpy().decode("utf-8"))
