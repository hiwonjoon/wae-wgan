import wae

if __name__ == "__main__":
    wae.run_mnist('_log/cat-wae-wgan-1norm/',int(1e5),100,500,z_dim=5)
