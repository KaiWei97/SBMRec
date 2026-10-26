import os
import pickle
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from txt_extractor import generate_text, txt_extractor, load_content
from img_extractor import img_extractor
from get_df import get_df
from tqdm import tqdm

def amazon(args):
    data_path = f'./dataset/reviews_{args.dataset}_5.json.gz'
    data_df = get_df(data_path)
    inter_df = data_df.rename(
        columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'timestamp', 'overall': 'stars'})
    inter_df = inter_df[['user', 'item', 'timestamp']]
    user_list = sorted(inter_df['user'].unique())
    user2id = dict(zip(user_list, range(1, len(user_list) + 1)))
    inter_df['user'] = inter_df['user'].apply(lambda x: user2id[x])
    return inter_df

def inter2txt(inter_df, txt_path):
    df = inter_df.sort_values(by=['user', 'timestamp'], kind='mergesort').reset_index(drop=True)
    with open(txt_path, 'w') as f:
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        for i, row in tqdm(df.iterrows(), desc='Generating inter file', total=df.shape[0]):
            user, item, t = row['user'], row['item'], row['timestamp']
            f.write('{}\t{}\t{}\n'.format(user, item, t))

def recbole2local(config, dataloader, local_path):
    uid_list, seq, target, interval, length = [], [], [], [], []
    user_field = config["USER_ID_FIELD"]
    seq_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    target_field = config["ITEM_ID_FIELD"]
    length_field = config["ITEM_LIST_LENGTH_FIELD"]
    interval_field = config["TIME_FIELD"] + config["LIST_SUFFIX"]
    for _, interaction in enumerate(dataloader):
        uid_list.append(interaction[user_field].long())
        seq.append(interaction[seq_field].long())
        target.append(interaction[target_field].long())
        interval.append(interaction[interval_field].long())
        length.append(interaction[length_field].long())
    uid_list, seq, target, interval, length = torch.cat(uid_list, dim=0), torch.cat(seq, dim=0), torch.cat(target, dim=0), torch.cat(interval, dim=0), torch.cat(length, dim=0)
    with open(local_path, 'wb') as f: pickle.dump((uid_list, seq, target, interval, length), f)

# the dataloaders of training and testing from recbole are unexpectedly different
def recbole2local_val(config, dataloader, local_path):
    uid_list, seq, target, interval, length = [], [], [], [], []
    user_field = config["USER_ID_FIELD"]
    seq_field = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    target_field = config["ITEM_ID_FIELD"]
    length_field = config["ITEM_LIST_LENGTH_FIELD"]
    interval_field = config["TIME_FIELD"] + config["LIST_SUFFIX"]
    for _, inter in enumerate(dataloader):
        interaction = inter[0]
        uid_list.append(interaction[user_field].long())
        seq.append(interaction[seq_field].long())
        target.append(interaction[target_field].long())
        interval.append(interaction[interval_field].long())
        length.append(interaction[length_field].long())
        length.append(interaction[length_field].long())
    uid_list, seq, target, interval, length = torch.cat(uid_list, dim=0), torch.cat(seq, dim=0), torch.cat(target, dim=0), torch.cat(interval, dim=0), torch.cat(length, dim=0)
    with open(local_path, 'wb') as f: pickle.dump((uid_list, seq, target, interval, length), f)

def prepare_inter(args):
    if not os.path.exists(args.inter_path):
        inter_df = amazon(args)
        inter2txt(inter_df, args.inter_path)

def prepare_txt_emb(args):
    if not os.path.exists(args.txt_emb):
        with open(args.item_dict_path, 'rb') as f: item2id = pickle.load(f)
        if not os.path.exists(args.txt_path):
            item_sentences = generate_text(args, item2id.keys(), ['title', 'categories', 'brand']) # omit category for future use
        else: item_sentences = load_content(args)
        item_sentences = sorted(item_sentences, key=lambda x: item2id[x[0]])
        sentences = [s[1] for s in item_sentences]
        txt_extractor(sentences, args)

def prepare_img_emb(args):
    with open(args.item_dict_path, 'rb') as f: item2id = pickle.load(f)
    item2id.pop('[PAD]')
    if not os.path.exists(args.img_emb):
        item_id_list = list(item2id.keys())
        item_id_list = sorted(item_id_list, key=lambda x: item2id[x])
        img_extractor(args, item_id_list)

def local_timestamp(args, data_path):
    with open(args.data_path.replace('00_seq', args.dataset + '.inter'), 'r') as f:
        line = f.readlines()[1:]
    tmp = [-1, 0]
    max_interval = 0
    for l in line:
        user, _, timestamp = l.strip().split('\t')
        user, timestamp = int(user), int(timestamp)
        if user != tmp[0]:
            tmp = [user, timestamp]
        else:
            interval = timestamp - tmp[1]
            tmp[1] = timestamp
            max_interval = max(interval, max_interval)
    max_interval = torch.log2(torch.tensor(max_interval) + 1).item()
    with open(data_path, 'wb') as f:
        print(max_interval)
        pickle.dump(max_interval, f)

def local_minmax_day(args, data_path):
    with open(args.data_path.replace('00_seq', args.dataset + '.inter'), 'r') as f:
        line = f.readlines()[1:]
    min_date, max_date = 9999999, 0
    for l in line:
        user, _, timestamp = l.strip().split('\t')
        user, timestamp = int(user), int(timestamp)
        min_date = min(int(timestamp / 86400), min_date)
        max_date = max(int(timestamp / 86400), max_date)
    with open(data_path, 'wb') as f:
        print(min_date)
        print(max_date)
        pickle.dump((min_date, max_date), f)

def prepare_seq(args):
    if not os.path.exists(args.data_path.replace('00', 'train')):
        config = Config(model='SASRec', dataset=f'./dataset/{args.dataset}/{args.dataset}', config_file_list=['./config/data.yaml'])
        dataset = create_dataset(config)
        train_data, dev_data, test_data = data_preparation(config, dataset)
        # recbole对物品进行了重新映射，因此需要将新的映射改写到原来的item2id中
        item2id = train_data.dataset.field2token_id['item_id']
        with open(args.item_dict_path, 'wb') as f: pickle.dump(item2id, f)
        train_data.shuffle = False
        recbole2local(config, train_data, args.data_path.replace('00', 'train'))
        recbole2local_val(config, dev_data, args.data_path.replace('00', 'dev'))
        recbole2local_val(config, test_data, args.data_path.replace('00', 'test'))
    if not os.path.exists(args.data_path.replace('00_seq', 'interval_num')):
        local_timestamp(args, args.data_path.replace('00_seq', 'interval_num'))
    if not os.path.exists(args.data_path.replace('00_seq', 'minmax_day')):
        local_minmax_day(args, args.data_path.replace('00_seq', 'minmax_num'))

def prepare_category(args, padding_idx=0):
    if not os.path.exists(args.data_path.replace('00_seq', 'cat.pt')):
        with open(args.item_dict_path, 'rb') as f: item2id = pickle.load(f)
        data_df = get_df(args.meta_path)
        cat_dict = {}
        cat_type_dict = {}
        for i in range(data_df.shape[0]):
            item = data_df['asin'][i]
            if item in item2id.keys():
                item_id = item2id[item]
                category = data_df['categories'][i][0]
                category_id = []
                for c in category:
                    if c not in cat_type_dict.keys():
                        cat_type_dict[c] = len(cat_type_dict)
                    category_id.append(cat_type_dict[c])
                category_id = torch.tensor(category_id)
                cat_dict[item_id] = category_id
        result = []
        for i in range(1, len(item2id)):
            ht = torch.nn.functional.one_hot(cat_dict[i], num_classes=len(cat_type_dict)).sum(dim=0).view(1, -1)
            result.append(ht)
        result.insert(padding_idx, torch.zeros((1, len(cat_type_dict)), dtype=torch.long))
        result = torch.cat(result, dim=0)
        torch.save(result, args.data_path.replace('00_seq', 'cat.pt'))
