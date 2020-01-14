import numpy as np
from metal.initializers.scheduler_init import SchedulerInitializer
import unittest


class Schedulerstest(unittest.TestCase):
    def test_constant_SchedulerInitializer_str(self):
        fcs = {'id': 'ConstantScheduler', 'lr': 0.005}
        cs = SchedulerInitializer('ConstantScheduler(lr=.005)')()
        assert (cs.hyperparameters == fcs)
        fcslr = {'id': 'ConstantScheduler', 'lr': 0.009}
        cslr = SchedulerInitializer(lr=.009)()
        assert (cslr.hyperparameters == fcslr)
