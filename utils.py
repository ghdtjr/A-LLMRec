import os
from datetime import datetime
from pytz import timezone

def log_args_time(args):
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
        
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'logs.txt'), 'a') as f:
        now = datetime.now(timezone('Asia/Seoul'))
        f.write("Timestamp: {}_{}-{}_{} \n".format(now.month, now.day, now.hour, now.minute))
        f.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
        f.write("\n--------------------------------------------------------")
    f.close()

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths

    
    
    