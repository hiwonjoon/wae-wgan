import wae
import wae_mmd

if __name__ == "__main__":
    #wae.run_mnist('_log/wae-wgan-1norm/',int(1e5),100,500,z_dim=5)
    #wae.run_celeba('_log/celeba/',int(1e5),10,200)

    wae_mmd.run_mnist('_log/mnist',int(1e4),10,200,num_iter=int(1e5))
