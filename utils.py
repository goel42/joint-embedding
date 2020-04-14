def print_hyperparams_tb(writer, args, train_dataset_size, test_dataset_size):
    writer.add_text('Dataset', 'Training set size: ' + str(train_dataset_size))
    writer.add_text('Dataset', 'Test set size: ' + str(test_dataset_size))

    writer.add_text('Hyperparams', 'epochs: ' + str(args.epochs))
    writer.add_text('Hyperparams', 'batch size: ' + str(args.train_batch_size))
    writer.add_text('Hyperparams', 'learning rate: ' + str(args.lr))
    writer.add_text('Hyperparams', 'margin: ' + str(args.margin))
    writer.add_text('Hyperparams', 'optimizer: ' + str(args.optimizer))