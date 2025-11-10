from perform_experiment import perform_experiment

n_train_images = 60_000
n_test_images = 10_000
run = lambda n_samples,\
             K_tr_tr_path,\
             K_ts_tr_path:\
            perform_experiment(
                n_samples, n_train_images, n_test_images,
                10, 3, 252,
                "./data/download/train.bin",
                "./data/download/test.bin",
                "./data/tmp/paulis.bin",
                K_tr_tr_path, K_ts_tr_path
            )


for i in range(1):
    n_samples = 4 * (2**i)
    run(n_samples,
        f"./data/out/K_train_train_{n_samples}_samples_{n_train_images}x{n_train_images}",
        f"./data/out/K_test_train_{n_samples}_samples_{n_test_images}x{n_train_images}")