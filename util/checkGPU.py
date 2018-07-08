import os

from keras.backend.tensorflow_backend import _is_current_explicit_device, _get_available_gpus

onGPU = not _is_current_explicit_device('CPU') and len(_get_available_gpus()) > 0

countGPU = os.environ.get('CUDA_VISIBLE_DEVICES', None)
