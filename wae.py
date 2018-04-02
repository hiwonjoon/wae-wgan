import tensorflow as tf
from commons.net import Net

class WAE_GAN(Net):
    pass

class WAE_WGAN(Net):
    def __init__(self,
                 x,
                 p_z,
                 Q_arch, #Encoder
                 G_arch, #Decoder
                 D_arch, #Discriminator
                 D_lambda, #Lambda for gradient_penalty for improved WGAN
                 c_fn, #l2_norm, etc.
                 backward_params,
                 param_scope):

        with tf.variable_scope(param_scope):
            with tf.variable_scope('Q') as phi_scope:
                q_net_spec,q_net_weights = Q_arch()
            with tf.variable_scope('G') as theta_scope:
                g_net_spec,g_net_weights = G_arch()
            with tf.variable_scope('D') as gamma_scope:
                d_net_spec,d_net_weights = D_arch()

        def _build_net(spec,_t):
            for block in spec:
                _t = block(_t)
            return _t

        summaries = []
        with tf.variable_scope('forward'):
            batch_size = tf.shape(x)[0]

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
                e = tf.random_uniform((batch_size,1), minval=0.0,maxval=1.0)
                z_hat = e * z + (1.0-e) * z_tilde

                D_z_tilde = tf.squeeze(_build_net(d_net_spec,z_tilde),axis=1) # (B,1) -> (B,)
                D_z = tf.squeeze(_build_net(d_net_spec,z),axis=1)
                D_z_hat = tf.squeeze(_build_net(d_net_spec,z_hat),axis=1)

                # calculage Loss
                critic_loss = \
                    tf.reduce_mean(D_z_tilde - D_z, axis=0)
                grad = tf.reshape( tf.gradients(D_z_hat,[z_hat])[0], (batch_size,-1) )
                gradient_penalty = \
                    tf.reduce_mean(
                        (tf.norm(grad,axis=1)-1.0)**2,
                        axis=0)

                L_D = critic_loss + D_lambda * gradient_penalty

            # TF Summary to observe learning statistics...
            summaries.append(tf.summary.scalar('recon_loss',L_recon))
            summaries.append(tf.summary.scalar('critic_loss',critic_loss))
            summaries.append(tf.summary.scalar('gradient_penalty',gradient_penalty))
            summaries.append(tf.summary.scalar('L_D_loss',L_D))

        if( backward_params is not None ):
            with tf.variable_scope('backward'):
                lr = backward_params['lr']
                lamb = backward_params['lambda']

                d_optimizer = tf.train.AdamOptimizer(lr)
                q_g_optimizer = tf.train.AdamOptimizer(lr)

                batchnorm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,disc_scope.name)
                assert len(batchnorm_ops) == 0, 'D_gamma should not have batch norm, use layer norm of instance norm'
                update_gamma = d_optimizer.minimize(L_D,var_list=tf.trainable_variables(gamma_scope.name))

                batchnorm_ops = \
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS,enc_scope.name)+\
                    tf.get_collection(tf.GraphKeys.UPDATE_OPS,dec_scope.name)

                print('batchnorm update ops for Q_phi(encoder):',batchnorm_ops)
                with tf.control_dependencies(batchnorm_ops):
                    update_phi_theta = \
                        q_g_optimizer.minimize(-1.*lamb*critic_loss + L_recon,
                                               var_list=tf.trainable_variables(phi_scope.name)+tf.trainable_variables(theta_scope.name))
            self.gp = gradient_penalty
            self.update_gamma = update_gamma
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

class WAE_MMD(Net):
    pass


########################
# Sample Running Script
########################

def run(num_iter,n_critic,model,ds,log_dir,summary_period,save_period,im_summary_period,**kwargs):
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
            for _ in range(n_critic):
                sess.run(model.update_gamma)

            _, summary_str = sess.run([model.update_phi_theta,model.summary])

            if( it % summary_period == 0 ):
                #tqdm.write('[%d]'%(it))
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
    batch_size = 64,
    n_critic = 10,
    D_lambda = 5,
    lr = 0.001,
    z_dim = 10,
    c_fn_type='l1',
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
    D_arch = partial(arch.fc_arch,
                     input_shape=(p_z_length,), # shape when flattened.
                     output_size=1,
                     num_layers=3,
                     embed_size=64,
                     act_fn='ELU-like')

    with tf.variable_scope('param_scope') as scope:
        # To clearly seperate the parameters belong to layers from tf ops.
        # Make it easier to reuse
        pass

    if c_fn_type == 'l2':
        c_fn = lambda x,y: tf.reduce_sum(tf.abs(x-y),axis=(1,2,3)) #use l1_distance for recon loss
    else:
        c_fn = lambda x,y: tf.reduce_sum((x-y)**2,axis=(1,2,3)) #use l2_distance for recon loss


    model = \
        WAE_WGAN(x,
                 p_z,
                 Q_arch,
                 G_arch,
                 D_arch,
                 D_lambda,
                 c_fn,
                 {'lr':lr, 'lambda':10.},
                 scope)

    run(**locals())


def run_celeba(
    log_dir,
    save_period,
    summary_period,
    im_summary_period,
    num_iter = int(1e6),
    batch_size = 64,
    n_critic = 10,
    D_lambda = 5,
    lr = 0.001,
    z_dim = 64,
    c_fn_type='l1'
):
    ds = dataset.CelebA(batch_size)
    x,_ = ds.train_data_op

    p_z = arch.Gaussian_P_Z(z_dim)
    p_z_length = p_z.length

    Q_arch = partial(arch.enc_with_bn_arch,
                     input_shape=(64,64,3),
                     output_size=p_z_length,
                     channel_nums=[128,256,512,1024])
    G_arch = partial(arch.dec_with_bn_arch,
                     input_size=p_z_length,
                     next_shape=(8,8,1024),
                     output_channel_num=3,
                     channel_nums=[512,256,128])
    D_arch = partial(arch.fc_arch,
                     input_shape=(p_z_length,), # shape when flattened.
                     output_size=1,
                     num_layers=4,
                     embed_size=512,
                     act_fn='ELU-like')

    with tf.variable_scope('param_scope') as scope:
        # To clearly seperate the parameters belong to layers from tf ops.
        # Make it easier to reuse
        pass

    if c_fn_type == 'l2':
        c_fn = lambda x,y: tf.reduce_sum(tf.abs(x-y),axis=(1,2,3)) #use l1_distance for recon loss
    else:
        c_fn = lambda x,y: tf.reduce_sum((x-y)**2,axis=(1,2,3)) #use l2_distance for recon loss

    model = \
        WAE_WGAN(x,
                 p_z,
                 Q_arch,
                 G_arch,
                 D_arch,
                 D_lambda,
                 c_fn,
                 {'lr':lr, 'lambda':10.},
                 scope)

    run(**locals())
