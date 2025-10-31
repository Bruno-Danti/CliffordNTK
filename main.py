from perform_experiment import perform_experiment

perform_experiment(
    10, 20034, 10000,
    
    10, 3, 252,
    "./data/download/train.bin",
    "./data/download/test.bin",
    "./data/tmp/paulis.bin",
    
    "./data/out/K_train_train_1.csv", "./data/out/K_test_train_1.csv"
)