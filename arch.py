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

