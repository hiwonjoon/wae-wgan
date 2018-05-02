import tensorflow as tf
from commons.net import Net

class WAE_MMD(Net):
    def __init__(self,
                 x,
                 p_z,
                 Q_arch, #Encoder
                 G_arch, #Decoder
                 c_fn, #l2_norm, etc.
                 backward_params,
                 param_scope):

        with tf.variable_scope(param_scope):
            with tf.variable_scope('Q') as phi_scope:
                q_net_spec,q_net_weights = Q_arch()
            with tf.variable_scope('G') as theta_scope:
                g_net_spec,g_net_weights = G_arch()

        def _build_net(spec,_t):
            for block in spec:
                _t = block(_t)
            return _t

        summaries = []
        with tf.variable_scope('forward'):
            batch_size = tf.shape(x)[0]
            d_z = p_z.length

            x = x
            z = p_z.sample(batch_size)

            with tf.variable_scope('enc') as enc_scope:
                z_tilde = _build_net(q_net_spec,x)

            with tf.variable_scope('dec') as dec_scope:
                x_recon = g_z_tilde = tf.reshape(_build_net(g_net_spec,z_tilde),
                                                 tf.shape(x))
                x_sample = tf.reshape(_build_net(g_net_spec,z), # Just to observe the performance as generator
                                      tf.shape(x))

                L_recon = tf.reduce_mean(c_fn(x,g_z_tilde),axis=0)

            # Calculate D(z_tilde), D(z), D(z_hat) for Train Discriminator;
            # whether z~P(z) and z_hat~Q(Z|X) can be distinguished or not?
            # Trained D_gamma function will give a meaningful distance metric for D_z(Q_z,P_z)
            with tf.variable_scope('discriminate') as disc_scope:
                n = tf.cast(batch_size,tf.float32)
                C_base = 2.*d_z*1

                z_dot_z = tf.matmul(z,z,transpose_b=True) #[B,B} matrix where its (i,j) element is z_i \dot z_j.
                z_tilde_dot_z_tilde = tf.matmul(z_tilde,z_tilde,transpose_b=True)
                z_dot_z_tilde = tf.matmul(z,z_tilde,transpose_b=True)

                dist_z_z = \
                    (tf.expand_dims(tf.diag_part(z_dot_z),axis=1)\
                        + tf.expand_dims(tf.diag_part(z_dot_z),axis=0))\
                    - 2*z_dot_z
                dist_z_tilde_z_tilde = \
                    (tf.expand_dims(tf.diag_part(z_tilde_dot_z_tilde),axis=1)\
                        + tf.expand_dims(tf.diag_part(z_tilde_dot_z_tilde),axis=0))\
                    - 2*z_tilde_dot_z_tilde
                dist_z_z_tilde = \
                    (tf.expand_dims(tf.diag_part(z_dot_z),axis=1)\
                        + tf.expand_dims(tf.diag_part(z_tilde_dot_z_tilde),axis=0))\
                    - 2*z_dot_z_tilde

                L_D = 0.
                #with tf.control_dependencies([
                #    tf.assert_non_negative(dist_z_z),
                #    tf.assert_non_negative(dist_z_tilde_z_tilde),
                #    tf.assert_non_negative(dist_z_z_tilde)]):

                for scale in [1.0]:
                    C = tf.cast(C_base*scale,tf.float32)

                    k_z = \
                        C / (C + dist_z_z + 1e-8)
                    k_z_tilde = \
                        C / (C + dist_z_tilde_z_tilde + 1e-8)
                    k_z_z_tilde = \
                        C / (C + dist_z_z_tilde + 1e-8)

                    loss = 1/(n*(n-1))*tf.reduce_sum(k_z)\
                        + 1/(n*(n-1))*tf.reduce_sum(k_z_tilde)\
                        - 2/(n*n)*tf.reduce_sum(k_z_z_tilde)

                    L_D += loss

            # TF Summary to observe learning statistics...
            summaries.append(tf.summary.scalar('recon_loss',L_recon))
            summaries.append(tf.summary.scalar('L_D',L_D))

        if( backward_params is not None ):
            with tf.variable_scope('backward'):
                lr = backward_params['lr']
                lamb = backward_params['lambda']

                q_g_optimizer = tf.train.AdamOptimizer(lr)

                batchnorm_ops = \
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS,enc_scope.name)+\
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS,dec_scope.name)

                print('batchnorm update ops for Q_phi(encoder):',batchnorm_ops)
                with tf.control_dependencies(batchnorm_ops):
                    update_phi_theta = \
                        q_g_optimizer.minimize(lamb*L_D + L_recon,
                                               var_list=tf.trainable_variables(phi_scope.name)+tf.trainable_variables(theta_scope.name))
            self.update_phi_theta = update_phi_theta

            self.summary = tf.summary.merge(summaries)
            self.sample_image_summary = tf.summary.merge([
                tf.summary.image('original',x,max_outputs=5),
                tf.summary.image('recon',tf.clip_by_value(x_recon,0.,1.),max_outputs=5),
                tf.summary.image('sample',tf.clip_by_value(x_sample,0.,1.),max_outputs=5)])

        # Add Save & Load methods to the class.
        super().__init__(param_scope)

        self.x_recon = tf.cast(tf.clip_by_value(x_recon,0.,1.)*255,tf.uint8)
        self.x_sample = tf.cast(tf.clip_by_value(x_sample,0.,1.)*255,tf.uint8)
        self.z_tilde = z_tilde

        self.L_recon = L_recon
        self.L_D = L_D


def run(num_iter,model,ds,log_dir,summary_period,save_period,im_summary_period,**kwargs):
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Execute Training!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    sess.graph.finalize()
    summary_writer = tf.summary.FileWriter(log_dir,sess.graph)
    #summary_writer.add_summary(hparams_summary.eval(session=sess))

    # Graph Initailize
    sess.run(init_op)
    sess.run(ds.train_data_init_op)

    from tqdm import tqdm
    try:
        for it in tqdm(range(num_iter),dynamic_ncols=True):
            l_recon, l_d, _, summary_str = sess.run([model.L_recon,model.L_D,model.update_phi_theta,model.summary])

            if( it % summary_period == 0 ):
                tqdm.write('[%d]%f,%f'%(it,l_recon,l_d))
                summary_writer.add_summary(summary_str,it)
            if( it % im_summary_period == 0 ):
                summary_writer.add_summary(sess.run(model.sample_image_summary),it)
            if( it % save_period == 0 ):
                model.save(log_dir,it)
    except KeyboardInterrupt:
        model.save(log_dir)

from functools import partial
import arch
import dataset

def run_mnist(
    log_dir,
    save_period,
    summary_period,
    im_summary_period,
    num_iter = int(1e6),
    batch_size = 128,
    lr = 0.0001,
    lamb = 10,
    z_dim = 8,
    c_fn_type='l2_sum',
):

    ds = dataset.MNIST(batch_size)
    x,_ = ds.train_data_op

    p_z = arch.Gaussian_P_Z(z_dim)
    p_z_length = p_z.length

    Q_arch = partial(arch.fc_arch,
                     input_shape=(784,),
                     output_size=p_z_length,
                     num_layers=3,
                     embed_size=256)
    G_arch = partial(arch.fc_arch,
                     input_shape=(p_z_length,),
                     output_size=784, # # of generated pixels
                     num_layers=3,
                     embed_size=256)

    with tf.variable_scope('param_scope') as scope:
        # To clearly seperate the parameters belong to layers from tf ops.
        # Make it easier to reuse
        pass

    if c_fn_type == 'l1':
        c_fn = lambda x,y: tf.reduce_mean(tf.abs(x-y),axis=(1,2,3)) #use l1_distance for recon loss
    if c_fn_type == 'l1_sum':
        c_fn = lambda x,y: tf.reduce_sum(tf.abs(x-y),axis=(1,2,3)) #use l1_distance for recon loss
    elif c_fn_type == 'l2':
        c_fn = lambda x,y: tf.reduce_mean((x-y)**2,axis=(1,2,3)) #use l2_distance for recon loss
    elif c_fn_type == 'l2_sum':
        c_fn = lambda x,y: tf.reduce_sum((x-y)**2,axis=(1,2,3)) #use l2_distance for recon loss
    else:
        assert False, 'not supported cost type'

    model = \
        WAE_MMD(x,
                p_z,
                Q_arch,
                G_arch,
                c_fn,
                {'lr':lr, 'lambda':lamb},
                scope)

    run(**locals())
