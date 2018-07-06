import os
import signal
import sys
import traceback
from argparse import ArgumentParser

import keras.backend as K
from tensorflow.python import debug as tf_debug

from keras.callbacks import CSVLogger, TensorBoard, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

from .dataset_provider import DatasetProvider
from .model import Model
from .config import base_configuration
from .callbacks import LogMetrics, LogLearningRate, LogTimestamp, StopAfterTimedelta, StopWhenValLossExploding


class Training(object):
    def __init__(self,
                 training_label,
                 model_weights_path=None,
                 min_delta=1e-4,
                 min_lr=1e-7,
                 log_metrics_period=4,
                 explode_ratio=0.25,
                 explode_patience=2,
                 max_q_size=10,
                 workers=1,
                 verbose=2):
        self.training_label = training_label
        self.epochs = base_configuration["params"]["epochs"]
        self.time_limit = base_configuration["params"]["time_limit"]
        self.reduce_lr_factor = base_configuration["params"]["reduce_lr_factor"]
        self.reduce_lr_patience = base_configuration["params"]["reduce_lr_patience"]
        self.early_stopping_patience = base_configuration["params"]["early_stopping_patience"]
        self.model_weights_path = model_weights_path
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.log_metrics_period = log_metrics_period
        self.explode_ratio = explode_ratio
        self.explode_patience = explode_patience
        self.max_q_size = max_q_size
        self.workers = workers
        self.verbose = verbose

        if not ((self.epochs is None) ^ (self.time_limit is None)):
            raise ValueError("Either conf.epochs or conf.time_limit must be set, but not both!")

        if self.time_limit:
            self.epochs = sys.maxsize

        self.dataset_provider = DatasetProvider()
        base_configuration["vocab_size"] = self.dataset_provider.vocab_size

        self.init_result_dir()
        self.init_callbacks()
        self.model = Model()
        # TODO: self.write_param_config()

        self._stop_training = False

    def stop_training(self):
        self._stop_training = True
        try:
            self.keras_model.stop_training = True
        # Race condition: ImageCaptioningModel.build is not called yet
        except AttributeError:
            pass

    @property
    def keras_model(self):
        return self.model.keras_model

    def path_from_result_dir(self, *paths):
        return os.path.join(self.result_dir, *paths)

    def init_callbacks(self):
        log_lr = LogLearningRate()
        log_ts = LogTimestamp()
        log_metrics = LogMetrics(self.dataset_provider, period=self.log_metrics_period)

        CSV_FILENAME = 'metrics-log.csv'
        self.csv_filepath = self.path_from_result_dir(CSV_FILENAME)
        csv_logger = CSVLogger(filename=self.csv_filepath)

        CHECKPOINT_FILENAME = 'model-weights.hdf5'
        self.checkpoint_filepath = self.path_from_result_dir(CHECKPOINT_FILENAME)
        model_checkpoint = ModelCheckpoint(filepath=self.checkpoint_filepath,
                                           monitor='val_cider',
                                           mode='max',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           period=1,
                                           verbose=self.verbose)

        tensorboard = TensorBoard(log_dir=self.result_dir, write_graph=False)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      mode='min',
                                      epsilon=self.min_delta,
                                      factor=self.reduce_lr_factor,
                                      patience=self.reduce_lr_patience,
                                      min_lr=self.min_lr,
                                      verbose=self.verbose)

        earling_stopping = EarlyStopping(monitor='val_loss',
                                         mode='min',
                                         min_delta=self.min_delta,
                                         patience=self.early_stopping_patience,
                                         verbose=self.verbose)

        stop_after = StopAfterTimedelta(timedelta=self.time_limit,
                                        verbose=self.verbose)

        stop_when = StopWhenValLossExploding(ratio=self.explode_ratio,
                                             patience=self.explode_patience,
                                             verbose=self.verbose)

        # TODO Add LearningRateScheduler. Is it still needed?

        self.callbacks = [
            log_lr,  # Must be before tensorboard
            log_metrics,  # Must be before model_checkpoint and, tensorboard
            model_checkpoint,
            tensorboard,  # Must be before log_ts
            log_ts,  # Must be before csv_logger
            csv_logger,
            reduce_lr,  # Must be after csv_logger
            stop_when,  # Must be the third last
            earling_stopping,  # Must be the second last
            stop_after,  # Must be the last
        ]

    def init_result_dir(self):
        self.result_dir = os.path.join(self.dataset_provider.training_results_dir, self.training_label)

        CONFIG_FILENAME = "hyperparams-config.yaml"
        config_filepath = self.path_from_result_dir(CONFIG_FILENAME)
        if os.path.exists(config_filepath):
            raise ValueError("Training label {} exists!".format(
                self.training_label))

        os.makedirs(self.result_dir, exist_ok=True)

    def run(self, debug=False):
        if debug:
            sess = K.get_session()
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            K.set_session(sess)

        print("Building model..")
        self.model.build(self.dataset_provider.vocabs())
        if self.model_weights_path:
            print("Loading model weights from {}..".format(self.model_weights_path))
            self.keras_model.load_weights(self.model_weights_path)

        # self.model.build() is expensive so it increases the chance of a race condition. Checking self._stop_training
        # will minimize it (but it is still possible).
        if self._stop_training:
            self._stop_training = False
            return

        print("Training {} is starting..".format(self.training_label))
        self.keras_model.fit_generator(
            generator=self.dataset_provider.training_set(),
            steps_per_epoch=self.dataset_provider.training_steps,
            epochs=self.epochs,
            validation_data=self.dataset_provider.validation_set(),
            validation_steps=self.dataset_provider.validation_steps,
            max_q_size=self.max_q_size,
            workers=self.workers,
            callbacks=self.callbacks,
            verbose=self.verbose)

        self._stop_training = False
        print("Training {} has finished.".format(self.training_label))


# class Checkpoint(object):
#     def __init__(self,
#                  new_training_label,
#                  training_dir,
#                  load_model_weights,
#                  log_metrics_period,
#                  config_override):
#         if 'epochs' in config_override and 'time_limit' in config_override:
#             raise ValueError('epochs and time_limit cannot be both passed!')
#         self.new_training_label = new_training_label
#         self.training_dir = training_dir
#         self.load_model_weights = load_model_weights
#         self.log_metrics_period = log_metrics_period
#         self.config_override = config_override
#
#     @property
#     def training(self):
#         training_dir = self.training_dir
#         hyperparam_path = os.path.join(training_dir, 'hyperparams-config.yaml')
#         model_weights_path = os.path.join(training_dir, 'model-weights.hdf5')
#
#         config_builder = config.FileConfigBuilder(hyperparam_path)
#         config_dict = config_builder.build_config()._asdict()
#         if self.config_override:
#             config_dict.update(self.config_override)
#             config_dict['time_limit'] = parse_timedelta(
#                                                     config_dict['time_limit'])
#             if 'epochs' in self.config_override:
#                 config_dict['time_limit'] = None
#             elif 'time_limit' in self.config_override:
#                 config_dict['epochs'] = None
#
#         conf = config.Config(**config_dict)
#         model_weights_path = (model_weights_path if self.load_model_weights
#                               else None)
#         return Training(training_label=self.new_training_label,
#                         conf=conf,
#                         model_weights_path=model_weights_path,
#                         log_metrics_period=self.log_metrics_period,
#                         explode_patience=sys.maxsize)

def main(training_label,
         training_dir=None,
         load_model_weights=False,
         log_metrics_period=4,
         unit_test=False,
         debug=False):
    model_weights_path = os.path.join(training_dir, 'model-weights.hdf5')
    model_weights_path = model_weights_path if load_model_weights else None
    if training_dir:
        training = Training(
            training_label=training_label,
            model_weights_path=model_weights_path,
            log_metrics_period=log_metrics_period,
            explode_patience=sys.maxsize)
    else:
        training = Training(training_label=training_label, log_metrics_period=log_metrics_period)

    def handler(signum, frame):
        print('Stopping training..')
        print('(Training will stop after the current epoch)')
        try:
            training.stop_training()
        except:
            traceback.print_exc(file=sys.stderr)
    signal.signal(signal.SIGINT, handler)

    training.run(debug)

    if unit_test:
        return training


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--training_label")
    parser.add_argument("--training_dir")
    parser.add_argument("--load_model_weights", action="store_true")
    parser.add_argument("--log_metrics_period")
    parser.add_argument("--unit_test", action="store_true")
    args = parser.parse_args()
    main(args.training_label,
         training_dir=args.training_dir,
         load_model_weights=args.load_model_weights,
         log_metrics_period=args.log_metrics_period,
         unit_test=args.unit_test)
