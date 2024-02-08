import os
import os.path
import gzip
from collections import defaultdict
from datetime import datetime
import json
from tqdm import tqdm

def parse(path):
    g = gzip.open(path, 'rb')
    for l in tqdm(g):
        yield json.loads(l)
        
def preprocess(fname):
    if os.path.isfile(f'./data/{fname}.txt') and os.path.isfile(f'./pre_train/sasrec/data/{fname}_review.json.gz'):
        return
    
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0
    
    data_type = 'amazon'
    dataset_name = fname

    if data_type == 'amazon':
        # file_path = f'./pre_train/sasrec/data/{fname}.json.gz'
        file_path = f'./data/{fname}.json.gz'
        f = open('./reviews_' + dataset_name + 'trash.txt', 'w')
        
        # counting interactions for each user and item
        for l in parse(file_path):
            line += 1
            f.write(" ".join([l['reviewerID'], l['asin'], str(l['overall']), str(l['unixReviewTime'])]) + ' \n')
            asin = l['asin']
            rev = l['reviewerID']
            time = l['unixReviewTime']
            countU[rev] += 1
            countP[asin] += 1
        f.close()
        
        usermap = dict()
        usernum = 0
        itemmap = dict()
        itemnum = 0
        User = dict()
        review_dict = {}
        name_dict = {'title':{}, 'description':{}}
        import json
        # f = open('./pre_train/sasrec/data/meta_Movies_and_TV.json', 'r')
        f = open(f'./data/meta_{fname}.json', 'r')
        json_data = f.readlines()
        f.close()
        data_list = [json.loads(line[:-1]) for line in json_data]
        meta_dict = {}
        for l in data_list:
            meta_dict[l['asin']] = l
        
        for l in parse(file_path):
            line += 1
            asin = l['asin']
            rev = l['reviewerID']
            time = l['unixReviewTime']
            
            # do not use item and user which interactions are less than 5.
            if countU[rev] < 5 or countP[asin] < 5:
                continue
            
            # userid를 1부터 다시 설정
            if rev in usermap:
                userid = usermap[rev]
            else:
                usernum += 1
                userid = usernum
                usermap[rev] = userid
                User[userid] = []
            
            if asin in itemmap:
                itemid = itemmap[asin]
            else:
                itemnum += 1
                itemid = itemnum
                itemmap[asin] = itemid
            User[userid].append([time, itemid])
            
            
            if itemmap[asin] in review_dict:
                try:
                    review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
                except:
                    a = 0
                try:
                    review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
                except:
                    a=0
            else:
                review_dict[itemmap[asin]] = {'review': {}, 'summary':{}}
                try:
                    review_dict[itemmap[asin]]['review'][usermap[rev]] = l['reviewText']
                except:
                    a = 0
                try:
                    review_dict[itemmap[asin]]['summary'][usermap[rev]] = l['summary']
                except:
                    a = 0
            try:
                if len(meta_dict[asin]['description']) ==0:
                    name_dict['description'][itemmap[asin]] = 'Empty description'
                else:
                    name_dict['description'][itemmap[asin]] = meta_dict[asin]['description'][0]
                name_dict['title'][itemmap[asin]] = meta_dict[asin]['title']
            except:
                a =0
        import pickle
        # with open(f'./pre_train/sasrec/data/{fname}_review.json.gz', 'wb') as tf:
        with open(f'./data/{fname}_review.json.gz', 'wb') as tf:
            pickle.dump(review_dict, tf)
        # with open(f'./pre_train/sasrec/data/{fname}_meta.json.gz', 'wb') as tf:
        with open(f'./data/{fname}_meta.json.gz', 'wb') as tf:
            pickle.dump(name_dict, tf)
        # with gzip.open(f'./pre_train/sasrec/data/{fname}_meta.json.gz', 'w') as f:
        #     json.dump(review_dict, f, indent=4)
        
        for userid in User.keys():
            User[userid].sort(key=lambda x: x[0])
            
        print(usernum, itemnum)
        
        f = open(f'./data/{fname}.txt', 'w')
        for user in User.keys():
            for i in User[user]:
                f.write('%d %d\n' % (user, i[1]))
        f.close()
        