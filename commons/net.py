import tensorflow as tf

class Net(object):
    def __init__(self,param_scope):
        def _sanitize_var_name(var):
            base_scope_name = param_scope.name.split('/')[-1]
            return ('/'.join(var.name.split(base_scope_name)[1:])).split(':')[0]

        save_vars = {_sanitize_var_name(var) : var for var in
                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,param_scope.name) }
        print('save vars:')
        for v in sorted(save_vars.keys()):
            print(v)

        self.saver = tf.train.Saver(var_list=save_vars,max_to_keep = 0)

    def save(self,dir,step=None):
        sess = tf.get_default_session()
        if(step is not None):
            self.saver.save(sess,dir+'/model.ckpt',global_step=step)
        else :
            self.saver.save(sess,dir+'/last.ckpt')

    def load(self,model):
        sess = tf.get_default_session()
        self.saver.restore(sess,model)

    @staticmethod
    def _build_net(spec,_t):
        for block in spec:
            _t = block(_t)
        return _t

