import os

from keras.backend.tensorflow_backend import _is_current_explicit_device, _get_available_gpus

on_GPU = not _is_current_explicit_device('CPU') and len(_get_available_gpus()) > 0

gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', None)

if gpu_list:
    gpu_list = list(range(0, len(gpu_list)))
else:
    gpu_list = []
