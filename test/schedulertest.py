import numpy as np
from metal.initializers.scheduler_init import SchedulerInitializer, ConstantScheduler
import unittest


class Schedulerstest(unittest.TestCase):
    def test_constant_SchedulerInitializer_str(self):
        fcs = {'id': 'ConstantScheduler', 'lr': 0.005}
        cs = SchedulerInitializer('ConstantScheduler(lr=.005)')()
        assert (cs.hyperparameters == fcs)
        fcslr = {'id': 'ConstantScheduler', 'lr': 0.009}
        cslr = SchedulerInitializer(lr=.009)()
        assert (cslr.hyperparameters == fcslr)

    def test_constant_SchedulerInitializer_dict(self):
        fcs = {'hyperparameters':{'id': 'ConstantScheduler', 'lr': 0.0055}}
        cs = SchedulerInitializer(fcs)()
        assert(cs.hyperparameters=={'id': 'ConstantScheduler', 'lr': 0.0055})

    def test_constant_SchedulerInitializer(self):
        v={'id': 'ConstantScheduler', 'lr': 0.101}
        cs = SchedulerInitializer(ConstantScheduler(lr=.101))()
        assert(cs.hyperparameters==v)
