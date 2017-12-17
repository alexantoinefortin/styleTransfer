import warnings, numpy as np
from keras import backend as K

class IncreaseBSOnPlateau:
    """Increase batch size when a metric has stopped improving.
    Models often benefit from increasing the batch size by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the batch size is increased until 'max_bs' is met,
    after which the learning rate is reduced.
    # Example
        ```python
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=0.001)
        model.fit(X_train, Y_train, callbacks=[reduce_lr])
        ```
    # Arguments
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, model, monitor='val_loss', factor_bs=2,
                 patience=10, verbose=0, mode='auto', epsilon=1e-4, cooldown=0,
                 max_bs=None, min_lr=0):
        #super(IncreaseBSOnPlateau, self).__init__()

        self.monitor = monitor
        if factor_bs <= 1.0:
            raise ValueError('IncreaseBSOnPlateau '
                             'does not support a factor_bs <= 1.0.')
        self.model = model
        self.factor_bs = float(factor_bs)
        self.factor_lr = 1.0 / self.factor_bs
        if type(max_bs)==type(None):# XXX
            raise ValueError('IncreaseBSOnPlateau '
                             'does not support a max_bs == None'
                             'Implementation notes: need to find how to do'
                             'model.history.params.samples/10')
        else:
            self.max_bs = max_bs
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Batch Size Plateau Increasing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0
        self.lr_epsilon = self.min_lr * 1e-4

    def on_train_begin(self, logs=None):
        self._reset()

    def update_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['bs'] = self.model.history.params.get('batch_size',None) #XXX: will break if .fit_generator
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        #current = logs.get(self.monitor)
        current = self.model.history.history[self.monitor][-1]
        print("current: {}".format(current))
        if current is None:
            warnings.warn(
                'Increase BS on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else: #AAF: current is a valid metric
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0
                new_bs = self.model.history.params.get('batch_size',None)
                new_lr = K.get_value(self.model.optimizer.lr)

            if self.monitor_op(current, self.best): #AAF: self.wait is the # of epoch since best results
                self.best = current
                self.wait = 0
                new_bs = self.model.history.params.get('batch_size',None)
                new_lr = K.get_value(self.model.optimizer.lr)
            elif not self.in_cooldown(): # current is not best
                if self.wait >= self.patience: # DO BS and LR calculation
                    old_bs = int(self.model.history.params.get('batch_size',None))
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if ((old_bs * self.factor_bs) > self.max_bs) and (old_bs < self.max_bs): # Can't increase BS by full margin
                        # g = lr * Samples / BS (Yi et al. 2017) g is the SGD noise
                        sample_size = int(self.model.history.params.get('samples',None))
                        new_bs = int(old_bs * self.factor_bs)
                        new_bs = min(new_bs, self.max_bs)
                        old_g = old_lr * sample_size / old_bs
                        new_g = old_g / self.factor_bs
                        new_lr = new_g * new_bs / self.model.history.params.get('samples')
                        if self.verbose > 0:
                            print('\nEpoch %05d: increasing batch size to %s and reducing lr to %s.' % (epoch + 1, new_bs, new_lr))
                    elif old_bs < self.max_bs: # Full increase in BS
                        new_bs = int(old_bs * self.factor_bs)
                        new_bs = min(new_bs, self.max_bs)
                        new_lr = old_lr
                        if self.verbose > 0:
                            print('\nEpoch %05d: increasing batch size to %s.' % (epoch + 1, new_bs))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                    elif old_lr > self.min_lr + self.lr_epsilon: # Reducing LR instead of increasing BS
                        new_lr = old_lr * self.factor_lr
                        new_lr = max(new_lr, self.min_lr)
                        new_bs = old_bs
                        #K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: reducing learning rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                    elif old_lr <= self.min_lr + self.lr_epsilon: # lr == min_lr
                        new_bs = old_bs
                        new_lr = old_lr
                else: # set bs and lr value while waiting
                    new_bs = self.model.history.params.get('batch_size',None)
                    new_lr = K.get_value(self.model.optimizer.lr)
                self.wait += 1
        return new_bs, new_lr

    def in_cooldown(self):
        return self.cooldown_counter > 0
