import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer, BertModel
from tqdm import tqdm
from fuzzy_match import match
from fuzzy_match import algorithims
from transformers import T5Tokenizer
import random
# macro
#ZUCO_SENTIMENT_LABELS = json.load(open('./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
#SST_SENTIMENT_LABELS = json.load(open('./dataset/stanfordsentiment/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, bert_tokenizer, 
                     eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False, test_input="noise", text_model=None):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []

        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)

        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []

        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])

        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    sentence_eeg_data = sent_obj['rawData']
    input_sample['target_string'] = target_string
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt',
                                return_attention_mask = True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]


    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    # try:
    #     sent_level_eeg_tensor = torch.from_numpy(sent_obj['sentence_level_EEG']) # This gives a dictionary
    # except:
    #     return None
    
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    # if sent_level_eeg_tensor.shape[1] < 30:
    #     return None
        # clean 0 length data

    input_sample['sent_level_EEG'] = sent_level_eeg_tensor
    #input_sample['sent_level_EEG'] = torch.randn(sent_level_eeg_tensor.size()) # random input code
    #print("NOISE:", input_sample['sent_level_EEG'])


    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    


    if len(sent_obj['word']) < len(target_string.split()) * 0.5:
        print(len(sent_obj['word']), len(target_string.split()), target_string)
        return None
    
    #if target_string in ZUCO_SENTIMENT_LABELS:
    #    input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    #else:
    #    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value
    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []
    input_sample["word_content"] = ""

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None

        word_embeddings.append(word_level_eeg_tensor)
        input_sample["word_content"] += (word['content'].replace('emp11111ty','empty').replace('film.1','film.') + " ")


    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))


    if test_input=='noise':
        rand_eeg = torch.randn(torch.stack(word_embeddings).size())
        input_sample['input_embeddings'] = rand_eeg # max_len * (105*num_bands)
        # input_sample['input_embeddings'] = torch.stack(word_embeddings)
        # print("rand_eeg:", rand_eeg)
        # print("input_embeddings:", input_sample['input_embeddings'].shape)
    else:
        # torch.rand(max_len, 840)  #为什么会影响后面的结果？？？会改变后面随机初始化的结果，训练过程对这个非常敏感
        input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)
        # print("EEG", input_sample['input_embeddings'])
    
    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None


    left_idx = 0
    left_list = [0]
    word_text_embeddings = []
    word_negative_embedding = []
    word_token_nums = []
    word_list = []
    for idx, word in enumerate(input_sample["word_content"].split()):
        right_idx = target_string.find(word, left_idx) + len(word)
        word_token_nums.append(len(tokenizer(target_string[left_idx:right_idx])['input_ids'][1:-1]))
        # label = tokenizer(target_string[left_list[max(0, idx-1)]:right_idx])['input_ids']
        # label = tokenizer(target_string[left_idx:right_idx])['input_ids']
        label = bert_tokenizer(target_string[0:right_idx])['input_ids']
        negative_labels = label.copy()
        for j in range(1, len(negative_labels)-1):
            if j == len(negative_labels)-2:
                negative_labels[j] = random.randint(0, 30521)
            else:
                if random.random() < 0.5:
                    negative_labels[j] = random.randint(0, 30521)
        input_ids = torch.tensor([label, negative_labels]).cuda()
        with torch.no_grad():
            embedding = text_model(input_ids=input_ids.cuda())['last_hidden_state'][:, -2, :]
        word_text_embeddings.append(embedding[0])
        word_negative_embedding.append(embedding[1])
        left_idx = right_idx
        left_list.append(left_idx)
        word_list.append(word)
    while len(word_text_embeddings) < max_len:
        word_text_embeddings.append(torch.ones(embedding.size(-1)).cuda())
        word_negative_embedding.append(torch.ones(embedding.size(-1)).cuda())
        word_token_nums.append(0)
    word_text_embeddings = torch.stack(word_text_embeddings)
    word_negative_embedding = torch.stack(word_negative_embedding)
    input_sample["word_text_embeddings"] = word_text_embeddings.cpu()
    input_sample["word_negative_embedding"] = word_negative_embedding.cpu()
    input_sample["word_token_nums"] = torch.tensor(word_token_nums)
    input_sample["word_list"] = "<DIV>".join(word_list)
    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, 
                 subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], 
                 setting = 'unique_sent', is_add_CLS_token = False, test_input='noise',
                 model_path = 'bert-base-uncased'):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        text_model = BertModel.from_pretrained(model_path).cuda()
        bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        text_model = text_model.eval()
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]
                print('[INFO]using subjects: ', subjects)
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            
            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')

            if setting == 'unique_sent':
                # take first 80% as trainset, 10% as dev and 10% as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer, bert_tokenizer,
                                                            eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, text_model=text_model)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer, bert_tokenizer,
                                                            eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, text_model=text_model)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider,total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer, bert_tokenizer,
                                                            eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, text_model=text_model)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW','ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKW', 'ZKB', 'ZKH','ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, text_model=text_model)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, text_model=text_model)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZDN']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token, test_input=test_input, text_model=text_model)
                            if input_sample is not None:
                                input_sample['subject'] = key
                                self.inputs.append(input_sample)
            print('++ adding task to dataset, now we have:', len(self.inputs))

        print('[INFO]input tensor size:', self.inputs[0]['input_embeddings'].size(), self.inputs[0]['word_content'])
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'], 
            input_sample['word_content'],
            input_sample["word_text_embeddings"],
            input_sample["word_token_nums"],
            input_sample["word_negative_embedding"],
            input_sample['subject'],
            input_sample["word_list"]
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 





