import sys
import argparse

help_format = lambda arg: f'{arg[2]} (default: {arg[1]}).'

def parser(arg_specs: dict[str, tuple]):
    parser = argparse.ArgumentParser()

    for arg, arg_type in arg_specs.items():
        help = help_format(arg_type)
        if arg_type[0] == bool:
            parser.add_argument(f'--{arg}', action=argparse.BooleanOptionalAction, help=help, default=arg_type[1])
        else:
            parser.add_argument(f'--{arg}', type=arg_type[0], help=help, default=arg_type[1])

    args = parser.parse_args()
    return vars(args)

def parse_cbow_args():
    arg_specs = {
        'window_size': (int, 3, 'Size of the context window (default: 3).'),
        'walk_length': (int, 0, 'Length of the random walk (minimum default: 0 = window_size).'),
        'num_walks': (int, 1000, 'Number of random walks to generate for each node (default: 1000).'),
        'embedding_dim': (int, 12, 'Dimension of the node embeddings (default: 12).'),
        'epochs': (int, 500, 'Number of epochs to train the model (default: 500).'),
        'batch_size': (int, 256, 'Batch size for training (default: 256).'),
        'verbose': (bool, False, 'Print the training and testing logs (default: False).'),
        'exec': (str, 'train', "Execution mode 'grid'|'train' (default: train)."),
    }

    if '--help' in sys.argv or '-h' in sys.argv:
        print('CBOW Model for Node Classification of Karate Club Graph\n')
        print('Usage: python training.py [options]\n')
        for arg, arg_type in arg_specs.items():
            print(f'\t--{arg}: \t{help_format(arg_type)}')
        print('\t-h,--help: \tPrint this message.')
        exit(0)

    args = parser(arg_specs)
    if len(args) == 0:
        return None

    return args

final_args = None
def parse_args():
    # Store and return parsed arguments 
    global final_args
    if final_args is not None:
        return final_args

    # Else parse the arguments
    final_args = parse_cbow_args()
    return final_args

def cbow_args():
    args = parse_args()
    # Return the CBOW arguments when grid search is not applied
    if args is None or args['exec'] == 'grid':
        return None

    return {
        'window_size': [args['window_size']],
        'walk_length': [args['walk_length']],
        'num_walks': [args['num_walks']],
        'embedding_dim': [args['embedding_dim']],
        'batch_size': [args['batch_size']],
    }