#!/usr/bin/env python

import logging
import os
from collections import OrderedDict
import sys

import numpy
import theano
from theano.tensor.type import TensorType

from blocks.algorithms import GradientDescent, Adam
from blocks.extensions import FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import PARAMETER
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
from fuel.transformers import Transformer

from picklable_itertools import cycle, imap

from utils import ShortPrinting, prepare_dir, load_df
from utils import AttributeDict
from nn import ApproxTestMonitoring, FinalTestMonitoring, TestMonitoring
from nn import LRDecay
from ladder import LadderAE

logger = logging.getLogger('main')


class Whitening(Transformer):
    """ Makes a copy of the examples in the underlying dataset and whitens it
        if necessary.
    """
    def __init__(self, data_stream, iteration_scheme, whiten, cnorm=None,
                 **kwargs):
        super(Whitening, self).__init__(data_stream,
                                        iteration_scheme=iteration_scheme,
                                        **kwargs)
        data = data_stream.get_data(slice(data_stream.dataset.num_examples))
        self.data = []
        for s, d in zip(self.sources, data):
            if 'features' == s:
                # Fuel provides Cifar in uint8, convert to float32
                d = numpy.require(d, dtype=numpy.float32)
                if cnorm is not None:
                    d = cnorm.apply(d)
                if whiten is not None:
                    d = whiten.apply(d)
                self.data += [d]
            elif 'targets' == s:
                d = unify_labels(d)
                self.data += [d]
            else:
                raise Exception("Unsupported Fuel target: %s" % s)

    def get_data(self, request=None):
        return (s[request] for s in self.data)


class SemiDataStream(Transformer):
    """ Combines two datastreams into one such that 'target' source (labels)
        is used only from the first one. The second one is renamed
        to avoid collision. Upon iteration, the first one is repeated until
        the second one depletes.
        """
    def __init__(self, data_stream_labeled, data_stream_unlabeled, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.ds_labeled = data_stream_labeled
        self.ds_unlabeled = data_stream_unlabeled
        # Rename the sources for clarity
        self.ds_labeled.sources = ('features_labeled', 'targets_labeled')
        # Rename the source for input pixels and hide its labels!
        self.ds_unlabeled.sources = ('features_unlabeled',)

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.ds_labeled.sources + self.ds_unlabeled.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.ds_labeled.close()
        self.ds_unlabeled.close()

    def reset(self):
        self.ds_labeled.reset()
        self.ds_unlabeled.reset()

    def next_epoch(self):
        self.ds_labeled.next_epoch()
        self.ds_unlabeled.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        unlabeled = self.ds_unlabeled.get_epoch_iterator(**kwargs)
        labeled = self.ds_labeled.get_epoch_iterator(**kwargs)
        assert type(labeled) == type(unlabeled)

        return imap(self.mergedicts, cycle(labeled), unlabeled)

    def mergedicts(self, x, y):
        return dict(list(x.items()) + list(y.items()))


def unify_labels(y):
    """ Work-around for Fuel bug where MNIST and Cifar-10
    datasets have different dimensionalities for the targets:
    e.g. (50000, 1) vs (60000,) """
    yshape = y.shape
    y = y.flatten()
    assert y.shape[0] == yshape[0]
    return y


def load_and_log_params(cli_params):
    cli_params = AttributeDict(cli_params)
    if cli_params.get('load_from'):
        p = load_df(cli_params.load_from, 'params').to_dict()[0]
        p = AttributeDict(p)
        for key in cli_params.iterkeys():
            if key not in p:
                p[key] = None
        new_params = cli_params
        loaded = True
    else:
        p = cli_params
        new_params = {}
        loaded = False

        # Make dseed seed unless specified explicitly
        if p.get('dseed') is None and p.get('seed') is not None:
            p['dseed'] = p['seed']

    logger.info('== COMMAND LINE ==')
    logger.info(' '.join(sys.argv))

    logger.info('== PARAMETERS ==')
    for k, v in p.iteritems():
        if new_params.get(k) is not None:
            p[k] = new_params[k]
            replace_str = "<- " + str(new_params.get(k))
        else:
            replace_str = ""
        logger.info(" {:20}: {:<20} {}".format(k, v, replace_str))
    return p, loaded


def make_datastream(dataset, indices, batch_size,
                    scheme=SequentialScheme):

    return SemiDataStream(
        data_stream_labeled=Whitening(
            DataStream(dataset),
            iteration_scheme=scheme(indices, batch_size),
            whiten=None, cnorm=None),
        data_stream_unlabeled=Whitening(
            DataStream(dataset),
            iteration_scheme=scheme(indices, batch_size),
            whiten=None, cnorm=None)
    )


def setup_model(p):
    ladder = LadderAE(p)
    # Setup inputs
    input_type = TensorType('float32', [False] * (1 + 1))
    x_only = input_type('features_unlabeled')
    x = input_type('features_labeled')
    y = theano.tensor.lvector('targets_labeled')
    ladder.apply(x, y, x_only)

    # Load parameters if requested
    if p.get('load_from'):
        with open(p.load_from + '/trained_params.npz') as f:
            loaded = numpy.load(f)
            cg = ComputationGraph([ladder.costs.total])
            current_params = VariableFilter(roles=[PARAMETER])(cg.variables)
            logger.info('Loading parameters: %s' % ', '.join(loaded.keys()))
            for param in current_params:
                assert param.get_value().shape == loaded[param.name].shape
                param.set_value(loaded[param.name])

    return ladder


def train_ladder(cli_params, dataset=None, save_to='results/ova_all_full'):
    cli_params['save_dir'] = prepare_dir(save_to)
    logfile = os.path.join(cli_params['save_dir'], 'log.txt')

    # Log also DEBUG to a file
    fh = logging.FileHandler(filename=logfile)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Logging into %s' % logfile)

    p, loaded = load_and_log_params(cli_params)

    ladder = setup_model(p)

    # Training
    all_params = ComputationGraph([ladder.costs.total]).parameters
    logger.info('Found the following parameters: %s' % str(all_params))

    # Fetch all batch normalization updates. They are in the clean path.
    bn_updates = ComputationGraph([ladder.costs.class_clean]).updates
    assert 'counter' in [u.name for u in bn_updates.keys()], \
        'No batch norm params in graph - the graph has been cut?'

    training_algorithm = GradientDescent(
        cost=ladder.costs.total, params=all_params,
        step_rule=Adam(learning_rate=ladder.lr))
    # In addition to actual training, also do BN variable approximations
    training_algorithm.add_updates(bn_updates)

    short_prints = {
        "train": {
            'T_C_class': ladder.costs.class_corr,
            'T_C_de': ladder.costs.denois.values(),
        },
        "valid_approx": OrderedDict([
            ('V_C_class', ladder.costs.class_clean),
            ('V_E', ladder.error.clean),
            ('V_C_de', ladder.costs.denois.values()),
        ]),
        "valid_final": OrderedDict([
            ('VF_C_class', ladder.costs.class_clean),
            ('VF_E', ladder.error.clean),
            ('VF_C_de', ladder.costs.denois.values()),
        ]),
    }

    ovadataset = dataset['ovadataset']
    train_indexes = dataset['train_indexes']
    val_indexes = dataset['val_indexes']

    main_loop = MainLoop(
        training_algorithm,
        # Datastream used for training
        make_datastream(ovadataset, train_indexes,
                        p.batch_size, scheme=ShuffledScheme),
        model=Model(ladder.costs.total),
        extensions=[
            FinishAfter(after_n_epochs=p.num_epochs),

            # This will estimate the validation error using
            # running average estimates of the batch normalization
            # parameters, mean and variance
            ApproxTestMonitoring(
                [ladder.costs.class_clean, ladder.error.clean] +
                ladder.costs.denois.values(),
                make_datastream(ovadataset, val_indexes,
                                p.batch_size),
                prefix="valid_approx"),

            # This Monitor is slower, but more accurate since it will first
            # estimate batch normalization parameters from training data and
            # then do another pass to calculate the validation error.
            FinalTestMonitoring(
                [ladder.costs.class_clean, ladder.error.clean_mc] +
                ladder.costs.denois.values(),
                make_datastream(ovadataset, train_indexes,
                                p.batch_size),
                make_datastream(ovadataset, val_indexes,
                                p.batch_size),
                prefix="valid_final",
                after_n_epochs=p.num_epochs),

            TrainingDataMonitoring(
                [ladder.costs.total, ladder.costs.class_corr,
                 training_algorithm.total_gradient_norm] +
                ladder.costs.denois.values(),
                prefix="train", after_epoch=True),

            ShortPrinting(short_prints),
            LRDecay(ladder.lr, p.num_epochs * p.lrate_decay, p.num_epochs,
                    after_epoch=True),
        ])
    main_loop.run()

    # Get results
    df = main_loop.log.to_dataframe()
    col = 'valid_final_error_matrix_cost'
    logger.info('%s %g' % (col, df[col].iloc[-1]))

    ds = make_datastream(ovadataset, val_indexes,
                         p.batch_size)
    outputs = ladder.act.clean.labeled.h[len(ladder.layers) - 1]
    outputreplacer = TestMonitoring()
    _, _, outputs = outputreplacer._get_bn_params(outputs)

    cg = ComputationGraph(outputs)
    f = cg.get_theano_function()

    it = ds.get_epoch_iterator(as_dict=True)
    res = []
    inputs = {'features_labeled': [],
              'targets_labeled': [],
              'features_unlabeled': []}
    # Loop over one epoch
    for d in it:
        # Store all inputs
        for k, v in d.iteritems():
            inputs[k] += [v]
        # Store outputs
        res += [f(*[d[str(inp)] for inp in cg.inputs])]

    # Concatenate all minibatches
    res = [numpy.vstack(minibatches) for minibatches in zip(*res)]
    inputs = {k: numpy.vstack(v) for k, v in inputs.iteritems()}

    if main_loop.log.status['epoch_interrupt_received']:
        return None
    return res[0], inputs
