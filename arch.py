from commons.ops import *
import numpy as np

ACT_FNS = {
    'ReLU': lambda t,**kwargs : tf.nn.relu(t),
    'ELU': lambda t,**kwargs : tf.nn.elu(t),
    'LeakyReLU': Lrelu,
    'ELU-like': lambda t,**kwargs : tf.nn.softplus(2*t+2)/2-1
}

def fc_arch(input_shape,output_size,num_layers,embed_size,act_fn='ReLU'):
    assert len(input_shape) == 1, input_shape
    input_size = input_shape[0]

    in_and_out = [(input_size,embed_size)]+\
                 [(embed_size,embed_size) for _ in range(num_layers-1)]

    net_spec = []
    for l,(in_,out_) in enumerate(in_and_out):
        net_spec += [
            Linear('linear_%d'%l,in_,out_),
            ACT_FNS[act_fn]
        ]
    net_spec.append(Linear('linear_%d'%(num_layers),embed_size,output_size))

    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights

def enc_with_bn_arch(input_shape,output_size,channel_nums,act_fn='ReLU'):
    assert len(input_shape) == 3, input_shape
    h,w,c = input_shape

    in_and_out = [(c,channel_nums[0])] + \
                 [(channel_nums[i],channel_nums[i+1]) for i in range(len(channel_nums)-1)]

    net_spec = []
    for l,(in_,out_) in enumerate(in_and_out):
        net_spec += [
            Conv2d('conv2d_%d'%(l),in_,out_,5,5,2,2,data_format='NHWC'),
            BatchNorm('conv2d_bn_%d'%(l),out_,axis=3),
            ACT_FNS[act_fn]
        ]
        h /= 2
        w /= 2

    net_spec.append(Linear('linear_%d'%(l+1),h*w*channel_nums[-1],output_size))

    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights

def dec_with_bn_arch(input_size,next_shape,output_channel_num,channel_nums,act_fn='ReLU'):
    assert len(next_shape) == 3, next_shape
    h,w,c = next_shape

    net_spec = [
        Linear('linear',input_size,h*w*c),
        lambda t, **kwargs: tf.reshape(t,[-1,h,w,c])]

    in_and_out = [(c,channel_nums[0])] + \
                 [(channel_nums[i],channel_nums[i+1]) for i in range(len(channel_nums)-1)]

    for l,(in_,out_) in enumerate(in_and_out):
        net_spec += [
            SymPadConv2d('sym_conv2d_%d'%(l),in_,out_,5,5),
            BatchNorm('sym_conv2d_bn_%d'%(l),out_,axis=3),
            ACT_FNS[act_fn]
        ]

    net_spec.append(Conv2d('conv2d_%d'%(l+1),channel_nums[-1],3,5,5,1,1,data_format='NHWC'))


    import types
    weights = [block.get_variables() if not isinstance(block,types.FunctionType) else {}
               for block in net_spec]
    return net_spec, weights


class P_Z(object):
    def __init__(self,length):
        self.length = length

    def sample(self,batch_size):
        """
        return tensor ops(vars) generating sample from P_z
        """
        raise NotImplemented

class Gaussian_P_Z(P_Z):
    def __init__(self,length):
        super().__init__(length)

    def sample(self,batch_size):
        return tf.random_normal([batch_size,self.length])

class Pseudo_P_Z(P_Z):
    def __init__(self,ph):
        super().__init__(ph.shape.as_list()[1])
        self.ph = ph

    def sample(self,batch_size):
        return self.ph

