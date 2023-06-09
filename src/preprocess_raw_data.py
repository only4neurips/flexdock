import sys
from utils.args import *

if __name__ == "__main__":
    from utils.db5_data import Unbound_Bound_Data
    args['data'] == 'db5'
    the_path = args['cache_path']

    if args['data'] == 'db5':
        raw_data_path= '/apdcephfs/share_1364275/kaithgao/flexdock_git/flexdock_0425/data/benchmark5.5/structures'
        split_files_path = '/apdcephfs/share_1364275/kaithgao/flexdock_git/flexdock_0425/data/benchmark5.5/cv/'
    else:
        assert args['data'] == 'dips'
        raw_data_path= './DIPS/data/DIPS/interim/pairs-pruned/' 
        split_files_path = './DIPS/data/DIPS/interim/pairs-pruned/'


    os.makedirs(the_path, exist_ok=True)  ## Directory may exist!

    num_splits = 1
    if args['data'] == 'db5':
        num_splits = 1


    for i in range(num_splits):
        print('\n\nProcessing split ', i)

        args['cache_path'] = os.path.join(the_path, 'cv_' + str(i))
        os.makedirs(args['cache_path'], exist_ok=True)

        if args['data'] == 'db5':
            split_files_path = os.path.join(split_files_path, 'cv_' + str(i))

        Unbound_Bound_Data(args, reload_mode='val', load_from_cache=False, raw_data_path=raw_data_path,
                           split_files_path=split_files_path, data_fraction=args['data_fraction'])
        
        Unbound_Bound_Data(args, reload_mode='train', load_from_cache=False, raw_data_path=raw_data_path,
                           split_files_path=split_files_path, data_fraction=args['data_fraction']) # Writes data into the cache folder.
    
        Unbound_Bound_Data(args, reload_mode='test', load_from_cache=False, raw_data_path=raw_data_path,
                           split_files_path=split_files_path, data_fraction=args['data_fraction'])
