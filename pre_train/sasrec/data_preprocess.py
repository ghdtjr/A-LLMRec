import os
import os.path
import gzip
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

def parse(path):
    g = gzip.open(path, 'rb')
    for l in tqdm(g):
        yield json.loads(l)
        
def preprocess(fname):
    countU = defaultdict(lambda: 0)
    countP = defaultdict(lambda: 0)
    line = 0

    file_path = f'../../data/amazon/{fname}.json.gz'
    
    # counting interactions for each user and item
    for l in parse(file_path):
        line += 1
        if ('Beauty' in fname) or ('Toys' in fname):
            if l['overall'] < 3:
                continue
        asin = l['asin']
        rev = l['reviewerID']
        time = l['unixReviewTime']

        countU[rev] += 1
        countP[asin] += 1
    
    usermap = dict()
    usernum = 0
    itemmap = dict()
    itemnum = 0
    User = dict()
    review_dict = {}
    name_dict = {'title':{}, 'description':{}}
    
    f = open(f'../../data/amazon/meta_{fname}.json', 'r')
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
        
        threshold = 5
        if ('Beauty' in fname) or ('Toys' in fname):
            threshold = 4
            
        if countU[rev] < threshold or countP[asin] < threshold:
            continue
        
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
                a = 0
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
    
    with open(f'../../data/amazon/{fname}_text_name_dict.json.gz', 'wb') as tf:
        pickle.dump(name_dict, tf)
    
    for userid in User.keys():
        User[userid].sort(key=lambda x: x[0])
        
    print(usernum, itemnum)
    
    f = open(f'../../data/amazon/{fname}.txt', 'w')
    for user in User.keys():
        for i in User[user]:
            f.write('%d %d\n' % (user, i[1]))
    f.close()
