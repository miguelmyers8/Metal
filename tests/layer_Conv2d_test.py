from metal.layers.conv2D import Conv2D
import unittest
import pytest
from autograd.tensor import Tensor
from autograd.parameter import Parameter
import numpy as np


true_c3out = np.array([[[[-0.79899037,  1.0320021 ,  0.23549888],
          [ 0.2674513 ,  0.9715072 ,  0.5355813 ],
          [ 0.3240632 , -0.00254388,  0.33129525]],

         [[-0.21604921,  0.48240316, -0.00326236],
          [-0.6017107 , -1.1812458 ,  0.16737136],
          [-0.72112125, -0.40808526, -0.32228673]]],


        [[[-0.4251591 ,  0.6834359 , -0.24868038],
          [-0.33330792,  0.03984545,  0.18328846],
          [ 0.49960452,  0.3543895 , -0.51435834]],

         [[ 1.5021228 , -0.75696313, -0.32508022],
          [ 1.0651203 , -0.8091068 , -0.24125697],
          [ 0.4442642 ,  0.8350314 ,  0.17313302]]]])

true_c3_wgrad = np.array([[[[-8.19585731e-01,  1.02086881e+00,  3.92410545e+00],
         [ 3.44848713e+00,  4.58107333e+00,  2.32733422e+00],
         [ 5.86168615e+00,  4.34923879e+00,  1.87556329e-01]],

        [[-4.78018040e+00, -2.71156708e+00,  4.28354817e-01],
         [-2.10215997e+00, -6.90309942e-01, -3.40407435e-01],
         [ 1.94318679e+00,  1.14058188e+00, -4.00548127e-01]],

        [[-2.89029148e+00, -2.59368381e+00, -2.28921285e-01],
         [-3.32849661e+00, -4.36327586e+00, -3.21610566e+00],
         [-2.08514161e+00, -3.93948311e+00, -3.95066029e+00]]],


       [[[ 1.49811499e-01, -2.13098314e-01,  1.12863103e+00],
         [ 8.92131329e-01,  1.30163425e-01,  1.64381389e+00],
         [ 2.57159119e+00,  2.28699271e+00,  9.89985327e-01]],

        [[-1.46197932e+00, -1.44890217e+00, -3.73661273e-01],
         [-2.29122984e+00, -2.36024265e+00, -3.83322042e-01],
         [ 2.48963954e-03,  6.79940372e-01,  7.14558677e-01]],

        [[-1.30307034e+00, -8.76438555e-01,  2.24176724e-02],
         [-2.03118774e+00, -2.27845384e+00, -6.28212379e-01],
         [-7.87470706e-01, -1.97420142e+00, -2.16523659e+00]]]])

true_c3_bgrad = np.array([[-20.11926958],
       [ -9.58107759]])
true_c4_bgrad = np.array([[18.],
       [18.]])
true_c4_wgrad = np.array([[[[ 1.43678437,  2.14247261,  3.43247878],
         [ 2.61229777,  3.13492292,  3.60126132],
         [ 2.1210093 ,  2.65681599,  1.8990049 ]],

        [[-0.5154294 , -0.91765755, -2.66714081],
         [-0.36534001, -0.91672178, -2.389348  ],
         [-1.3768536 , -1.59989276, -1.78644551]]],


       [[[ 1.43678437,  2.14247261,  3.43247878],
         [ 2.61229777,  3.13492292,  3.60126132],
         [ 2.1210093 ,  2.65681599,  1.8990049 ]],

        [[-0.5154294 , -0.91765755, -2.66714081],
         [-0.36534001, -0.91672178, -2.389348  ],
         [-1.3768536 , -1.59989276, -1.78644551]]]])




true_c4out = np.array([[[[-0.42370066, -0.43009187, -0.50845811],
         [ 0.25508575, -0.575298  , -0.10159585],
         [-0.06149733, -0.11085947,  0.06543428]],

        [[-0.22442041, -0.08442765, -0.51309691],
         [ 0.00406582, -0.6208603 , -0.24300702],
         [ 0.24924928,  0.03614051,  0.41039784]]],


       [[[ 0.34074468, -0.23730474, -0.03891834],
         [ 0.55066421, -0.45215798,  0.41334732],
         [-0.06129876, -0.34717051, -0.14316889]],

        [[-0.4970375 ,  0.30046468, -0.09504793],
         [-0.45896246, -0.37225861,  0.40149224],
         [-0.15349387, -0.37440159,  0.08442678]]]])

true_c3back = np.array([[[[ 1.03304266,  0.86942988,  0.52080743],
         [ 0.99516422,  1.37313643,  1.16460699],
         [ 0.83144443,  1.13726439,  0.7903138 ]],

        [[-0.70701887, -0.57932473, -0.20696652],
         [-0.39403503, -0.34673656,  0.14282782],
         [-0.0811552 , -0.15464339,  0.35380414]],

        [[ 0.68105897,  0.67337833,  0.38279948],
         [-0.14655006,  0.35528163,  0.43895762],
         [-1.06002084, -0.28564495,  0.38266378]]],


       [[[ 1.03304266,  0.86942988,  0.52080743],
         [ 0.99516422,  1.37313643,  1.16460699],
         [ 0.83144443,  1.13726439,  0.7903138 ]],

        [[-0.70701887, -0.57932473, -0.20696652],
         [-0.39403503, -0.34673656,  0.14282782],
         [-0.0811552 , -0.15464339,  0.35380414]],

        [[ 0.68105897,  0.67337833,  0.38279948],
         [-0.14655006,  0.35528163,  0.43895762],
         [-1.06002084, -0.28564495,  0.38266378]]]])

true_c4back = np.array([[[[-0.84184858, -1.18733238, -1.0427759 ],
         [-1.14329866, -1.80500468, -1.21498727],
         [-0.73538406, -1.43369579, -0.65530746]],

        [[-0.97806548, -1.17552217, -0.49294909],
         [-0.55965415, -0.83635785, -0.33272143],
         [-0.03646282, -0.23247876, -0.14632704]]],


       [[[-0.84184858, -1.18733238, -1.0427759 ],
         [-1.14329866, -1.80500468, -1.21498727],
         [-0.73538406, -1.43369579, -0.65530746]],

        [[-0.97806548, -1.17552217, -0.49294909],
         [-0.55965415, -0.83635785, -0.33272143],
         [-0.03646282, -0.23247876, -0.14632704]]]])

class TestLayerConv2d(unittest.TestCase):
    def test_conv2d_f(self):

        c3=Conv2D(n_filters=2, filter_shape=(3,3), stride=1, input_shape=(3,3,3), padding='same',seed=1)
        c4=Conv2D(n_filters=2, filter_shape=(3,3), stride=1,input_shape=c3.output_shape(), padding='same',seed=2)
        c3.initialize()
        c4.initialize()

        np.random.seed(4)
        x = np.random.randn(2,3,3,3)
        _x_ = Tensor(x,True)
        c3out = c3.forward_pass(_x_)
        c4out = c4.forward_pass(c3out)

        assert np.allclose(c3out.data, true_c3out)
        assert np.allclose(c4out.data, true_c4out)

    def test_conv2d_b(self):
        c3=Conv2D(n_filters=2, filter_shape=(3,3), stride=1, input_shape=(3,3,3), padding='same',seed=1)
        c4=Conv2D(n_filters=2, filter_shape=(3,3), stride=1,input_shape=c3.output_shape(), padding='same',seed=2)
        c3.initialize()
        c4.initialize()

        np.random.seed(4)
        x = np.random.randn(2,3,3,3)
        _x_ = Tensor(x,True)
        c3out = c3.forward_pass(_x_)
        c4out = c4.forward_pass(c3out)

        assert np.allclose(c3out.data, true_c3out)
        assert np.allclose(c4out.data, true_c4out)

        c4out.sum().backward()

        assert np.allclose(c3.INPUT.grad.data, true_c3back)
        assert np.allclose(c4.INPUT.grad.data, true_c4back)
        assert np.allclose(c3.w.grad.data, true_c3_wgrad)
        assert np.allclose(c4.w.grad.data, true_c4_wgrad)
        assert np.allclose(c3.b.grad.data, true_c3_bgrad)
        assert np.allclose(c4.b.grad.data, true_c4_bgrad)
