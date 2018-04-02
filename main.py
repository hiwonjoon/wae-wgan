import wae

if __name__ == "__main__":
    #wae.run_mnist('_log/wae-wgan-1norm/',int(1e5),100,500,z_dim=5)
    wae.run_celeba('_log/celeba/',int(1e5),10,200)
