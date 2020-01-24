import unittest
from metal.initializers.optimizer_init import OptimizerInitializer, Adam
import numpy as np
from numpy.testing import assert_almost_equal

adamingrad = np.array([[1.62434536, 0.,         0.        ],
                        [0.,         0.86540763, 0.        ]])

adamin = np.array([[ 1.62434536, -0.61175641, -0.52817175],
                        [-1.07296862,  0.86540763, -2.3015387 ]])

adamout_ = np.array([[ 1.62334536, -0.61175641, -0.52817175],
                        [-1.07296862,  0.86440763 ,-2.3015387 ]])


adamoutdic = np.array([[ 1.5353082,  -0.61175641, -0.52817175],
                        [-1.07296862,  0.78417979, -2.3015387 ]])

sgdout_ = np.array([[ 1.60810191, -0.61175641, -0.52817175],
                        [-1.07296862 , 0.85675355, -2.3015387 ]])

class optimizerstest(unittest.TestCase):
    def test_Adam_OptimizerInitializer_init_from_str(self):
        np.random.seed(1)
        scheduler_ = None
        adam = OptimizerInitializer('adam')()
        adamout = adam(adamin,adamingrad,'adam_test_1')
        assert('ConstantScheduler' == adam.lr_scheduler.__class__.__name__)
        assert np.allclose(adamout,adamout_)
        assert(adam.hyperparameters['id']=='Adam')
        assert(adam.hyperparameters['lr']==0.001)
        assert(adam.hyperparameters['decay1']==0.9)
        assert(adam.hyperparameters['decay2']==0.999)
        assert(adam.hyperparameters['eps']==1e-7)
        assert(adam.hyperparameters['clip_norm']==None)

    def test_Adam_OptimizerInitializer_init_from_dict(self):
        hp = {'hyperparameters':{'id':'Adam','lr':.1,'eps':.2,'decay1':.3,'decay2':.4,'clip_norm':None,'lr_scheduler': 'ConstantScheduler(lr=.1)'}}
        dic = OptimizerInitializer(hp)
        op2 = dic()
        assert (op2.hyperparameters == hp['hyperparameters'])
        adamout = op2(adamin,adamingrad,'adam_test_2')
        assert np.allclose(adamout,adamoutdic)

    def test_Adam_OptimizerInitializer_init_from_class(self):
        adam = OptimizerInitializer(Adam())()
        adamout = adam(adamin,adamingrad,'adam_test_1')
        assert('ConstantScheduler' == adam.lr_scheduler.__class__.__name__)
        assert np.allclose(adamout,adamout_)
        assert(adam.hyperparameters['id']=='Adam')
        assert(adam.hyperparameters['lr']==0.001)
        assert(adam.hyperparameters['decay1']==0.9)
        assert(adam.hyperparameters['decay2']==0.999)
        assert(adam.hyperparameters['eps']==1e-7)
        assert(adam.hyperparameters['clip_norm']==None)


    def test_Adam_parents(self):
        adam = Adam()
        adam.step()
        step = adam.cur_step
        adam.reset_step()
        stepr = adam.cur_step
        assert(step == 1)
        assert(stepr == 0 )
        assert(adam.hyperparameters == adam.copy().hyperparameters)


class optimizerstestsgd(unittest.TestCase):
    def test_sgd_OptimizerInitializer_init_from_str(self):
        np.random.seed(1)
        scheduler_ = None
        sgd = OptimizerInitializer('sgd')()
        adamout = sgd(adamin,adamingrad,'adam_test_1')
        assert('ConstantScheduler' == sgd.lr_scheduler.__class__.__name__)
        assert np.allclose(adamout,sgdout_)
        assert(sgd.hyperparameters['id']=='SGD')
        assert(sgd.hyperparameters['lr']==0.01)
        assert(sgd.hyperparameters['clip_norm']==None)
        assert(sgd.hyperparameters['lr_scheduler']=='ConstantScheduler(lr=0.01)')
