import random
import socket
import datetime
import time

import pycrayon
import numpy as np
import mxnet as mx

from collections import namedtuple

CrayonSettings = namedtuple('CrayonSettings', ['host', 'port'])
CRAYON_SETTINGS = CrayonSettings(host='localhost', port='8889')


def get_experiment(name, settings=CRAYON_SETTINGS):
    """Creates a pycrayon experiment object to log data to."""
    experiment_date = datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')
    experiment_name = '{dt}_{host};{name}'.format(
        dt=experiment_date,
        host=socket.gethostname(),
        name=name,
    )
    return get_crayon_client(settings=settings).create_experiment(experiment_name)


def get_crayon_client(settings=CRAYON_SETTINGS):
    return pycrayon.CrayonClient(hostname=settings.host, port=settings.port)


def clear_expts(settings=CRAYON_SETTINGS):
    get_crayon_client(settings=settings).remove_all_experiments()


def monitor_train_acc(param,disp = 100):
    print(param)
    if param.nbatch + 1 == num_batches:  # last batch
        metric = dict(param.eval_metric.get_name_value())
        for key in metric:
            expt.add_scalar_dict(
                {
                    'train_'+ key : metric[key],
                },
            )
       # time.sleep(0.001)




if __name__ == '__main__':
    clear_expts()
    expt = get_experiment(name='mlp')
    num_examples = 60000
    batch_size = 100
    num_batches = num_examples / batch_size
    print('1')
    train_iter = mx.io.MNISTIter(image='data/train-images-idx3-ubyte', label='data/train-labels-idx1-ubyte',
                                 batch_size=batch_size, flat=True)
    eval_iter = mx.io.MNISTIter(image='data/t10k-images-idx3-ubyte', label='data/t10k-labels-idx1-ubyte',
                                batch_size=batch_size, flat=True)
    print('2')
    num_classes = 10
    mlp = mx.sym.Variable('data')
    mlp = mx.sym.FullyConnected(data=mlp, name='fc1', num_hidden=128)
    mlp = mx.sym.Activation(data=mlp, name='relu1', act_type="relu")
    mlp = mx.sym.FullyConnected(data=mlp, name='fc2', num_hidden=64)
    mlp = mx.sym.Activation(data=mlp, name='relu2', act_type="relu")
    mlp = mx.sym.FullyConnected(data=mlp, name='fc3', num_hidden=num_classes)
    mlp = mx.sym.SoftmaxOutput(data=mlp, name='softmax')

    model = mx.mod.Module(symbol=mlp, context=mx.cpu())
    optimizer_params = {
        'learning_rate': 0.01,  # learning rate
        'momentum': 0.9,  # momentum
        'wd': 0.0001,  # weight decay
        'lr_scheduler': mx.lr_scheduler.FactorScheduler(num_batches, factor=0.9)
    }
    batch_end_callbacks = [mx.callback.Speedometer(batch_size, 100)]
    epoch_end_callbacks = []
    eval_batch_end_callbacks = []
    eval_end_callbacks = []

    eval_metrics = ['crossentropy', 'accuracy']
    top_k=5
    if top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))


    batch_end_callbacks.append(monitor_train_acc)


    def monitor_eval_acc(param):
        metric = dict(param.eval_metric.get_name_value())
        for key in metric:
            expt.add_scalar_dict(
                {
                    'eval_' + key: metric[key],
                },
            )

    eval_batch_end_callbacks.append(monitor_eval_acc)


    def monitor_fc1_gradient(g):
#        print('add')
        expt.add_histogram_value(
            name='affine_1_weights',
            hist=g.asnumpy().flatten().tolist(),
            tobuild=True
            # We could also manually set the step number.
        )
#        summary_writer.add_summary(tensorboard.summary.histogram('fc1-backward-weight', g.asnumpy().flatten()))
        stat = mx.nd.norm(g) / np.sqrt(g.size)
        return stat


    monitor = mx.mon.Monitor(100, monitor_fc1_gradient, pattern='fc1_backward_weight')


    # monitor fc1 weight every 100 batches
    def monitor_fc1_weight(param):
        if param.nbatch % 100 == 0:
            arg_params, aux_params = param.locals['self'].get_params()
            expt.add_histogram_value(
                name='fc1-weight',
                hist=arg_params['fc1_weight'].asnumpy().flatten().tolist(),
                tobuild=True
                # We could also manually set the step number.
            )

    batch_end_callbacks.append(monitor_fc1_weight)





    model.fit(
        train_data=train_iter,
        begin_epoch=0,
        num_epoch=20,
        eval_data=eval_iter,
        eval_metric=eval_metrics,
        optimizer='sgd',
        optimizer_params=optimizer_params,
        initializer=mx.init.Uniform(),
        batch_end_callback=batch_end_callbacks,
        epoch_end_callback=epoch_end_callbacks,
        eval_batch_end_callback=eval_batch_end_callbacks,
        eval_end_callback=eval_end_callbacks,
        monitor=monitor
    )

    expt.to_zip('result.zip')
#    mx.viz.plot_network(mlp)