import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
import torch.nn.functional as F
import time
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationDecoder
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge
from config import get_config
import evaluate
from evaluate import load
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, BertModel
from model import LLMTranslator
import bert_score
import random



metric = evaluate.load("sacrebleu")
cer_metric = load("cer")
wer_metric = load("wer")

def remove_text_after_token(text, token='</s>'):
    # 查找并删除</s>后面的文本
    token_index = text.find(token)
    if token_index != -1:  
        return text[:token_index] 
    return text 


def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = './results/temp.txt' , 
               score_results='./score_results/task.txt', input_type = 'EEG', pretrained_model=None, llama2Tokenizer=None, embedding_model=None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    start_time = time.time()
    model.eval()   # Set model to evaluate mode
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []


    num = 0
    subject_list = []
    s_num = 0
    subjeces = ['ZAB', 'ZDM', 'ZGW','ZDN', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH', 'example1', 'example2']
    embeddings = {}
    word_lists = {}
    for s in subjeces:
        word_lists[s] = []
        embeddings[s] = torch.ones((1, 840))
    
    with open(output_all_results_path,'w') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
            sentiment_labels, word_content, word_text_embeddings, word_token_nums, word_negative_embedding, subject, word_list in tqdm(dataloaders['test']):
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float() # B, 56, 840
            word_text_embeddings = word_text_embeddings.float().to(device)
            word_token_nums = word_token_nums.to(device)
            target_ids_batch = target_ids.to(device) # B, 56
            input_mask_invert_batch = input_mask_invert.to(device) # B, 56

            # 原来使用BERT token
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # 转为使用llama2 token
            target_ids_batch = llama2Tokenizer(target_string, return_tensors='pt')['input_ids'].to(device)
            target_tokens = llama2Tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            word_content = word_content[0].split()
            left_idx = target_string.find(word_content[0].lower()) + len(word_content[0])
            left_idx = len(llama2Tokenizer([target_string[:left_idx]])['input_ids'][0])
            num += 1

            encoder_embedding = input_embeddings_batch[~input_mask_invert_batch.bool()].detach().cpu()
            embeddings[subject[0]] = torch.cat((embeddings[subject[0]], encoder_embedding), dim=0)
            word_lists[subject[0]] += word_list[0].split("<DIV>")
            subject_list += [subject[0]] * encoder_embedding.shape[0]
            # print(len(word_list[0].split("<DIV>")), input_masks.float().sum().item())

            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)


            predictions = model.generate(input_embeddings_batch=input_embeddings_batch, 
                                                  input_mask_invert=input_mask_invert_batch, 
                                                  input_ids=target_ids_batch[:, :left_idx], 
                                                  word_token_nums=word_token_nums[0],
                                                  word_text_embeddings=word_text_embeddings, 
                                                  LLM=pretrained_model, 
                                                  embedding_model=embedding_model,
                                                  max_length=target_mask.sum().item())
            
            
            predictions = torch.squeeze(predictions)
            predicted_string = remove_text_after_token(llama2Tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>',''))
            print(target_string)
            print(predicted_string)
            f.write(f'predicted string with tf: {predicted_string}\n')
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != llama2Tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = llama2Tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            
            f.write(f'################################################\n\n\n')

    
    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    corpus_bleu_scores = []
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        corpus_bleu_scores.append(corpus_bleu_score)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)


    """ calculate sacre bleu score """
    
    reference_list = [[item] for item in target_string_list]


    sacre_blue = metric.compute(predictions=pred_string_list, references=reference_list)
    print("sacreblue score", sacre_blue)


    print()
    """ calculate rouge score """
    rouge = Rouge()

    try:
        rouge_scores = rouge.get_scores(pred_string_list, target_string_list, avg = True, ignore_empty=True)
        print("Rouge score:", rouge_scores)
    except ValueError as e:
        rouge_scores = 'Hypothesis is empty'
    print()


    bertscore_P, bertscore_R, bertscore_F1 = bert_score.score(pred_string_list, target_string_list, lang="en", verbose=True)
    bertscore_P = bertscore_P.mean().item()
    bertscore_R = bertscore_R.mean().item()
    bertscore_F1 = bertscore_F1.mean().item()
    print("Bert score:", bertscore_P, bertscore_R, bertscore_F1)

    print()
    """ calculate WER score """
    wer_scores = wer_metric.compute(predictions=pred_string_list, references=target_string_list)
    print("WER score:", wer_scores)
    

    """ calculate CER score """
    cer_scores = cer_metric.compute(predictions=pred_string_list, references=target_string_list)
    print("CER score:", cer_scores)


    end_time = time.time()
    print(f"Evaluation took {(end_time-start_time)/60} minutes to execute.")

     # score_results (only fix teacher-forcing)
    file_content = [
        f'corpus_bleu_score = {corpus_bleu_scores}',
        f'sacre_blue_score = {sacre_blue}',
        f'rouge_scores = {rouge_scores}',
        f'bert_score = ["p": {bertscore_P}, "r": {bertscore_R}, "f1": {bertscore_F1}]',
        f'wer_scores = {wer_scores}',
        f'cer_scores = {cer_scores}',
    ]
    
    with open(score_results, "a") as file_results:
        for line in file_content:
            if isinstance(line, list):
                for item in line:
                    file_results.write(str(item) + "\n")
            else:
                file_results.write(str(line) + "\n")


if __name__ == '__main__': 
    batch_size = 1
    ''' get args'''
    args = get_config('eval_decoding')
    test_input = args['test_input']
    print("test_input is:", test_input)
    train_input = args['train_input']
    print("train_input is:", train_input)
    ''' load training config'''
    training_config = json.load(open(args['config_path']))


    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'  # unique_subj unique_sent

    task_name = training_config['task_name']
    model_name = training_config['model_name']
    dataset_path = args['dataset_path']
    model_path = args['model_path']
    llm_path = args['llm_path']

    if test_input == 'EEG' and train_input=='EEG':
        print("EEG and EEG")
        output_all_results_path = f'./results/{task_name}-{model_name}-all_decoding_results.txt'
        score_results = f'./score_results/{task_name}-{model_name}.txt'
    else:
        output_all_results_path = f'./results/{task_name}-{model_name}-{train_input}_{test_input}-all_decoding_results.txt'
        score_results = f'./score_results/{task_name}-{model_name}-{train_input}_{test_input}.txt'


    ''' set random seeds '''
    seed_val = 888 #500
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = 0
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    # task_name = 'task1_task2_task3'

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = dataset_path + 'task1-SR/pickle/task1-SR-dataset.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = dataset_path + 'task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = dataset_path + 'task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = dataset_path + 'task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()
    
    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained(model_path)
    elif model_name == 'LLMTranslator':
        tokenizer = BertTokenizer.from_pretrained(model_path)

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, 
                            subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, 
                            setting = dataset_setting, test_input=test_input, model_path=model_path)
    
    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)
    dataloaders = {'test':test_dataloader}


    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained(model_path)
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    
    elif model_name == 'BrainTranslatorNaive':
        pretrained_bart = BartForConditionalGeneration.from_pretrained(model_path)
        model = BrainTranslatorNaive(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'BertGeneration':
        pretrained = BertGenerationDecoder.from_pretrained(model_path, is_decoder = True)
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
        
    
    elif model_name == 'LLMTranslator':
        pretrained = LlamaForCausalLM.from_pretrained(llm_path, device_map="auto")
        pretrained = pretrained.bfloat16().eval()
        embedding_model = BertModel.from_pretrained(model_path, device_map="auto")
        embedding_model = embedding_model.eval()
        llama2Tokenizer = LlamaTokenizer.from_pretrained(llm_path)
        model = LLMTranslator(in_feature = 105*len(bands_choice), eeg_encoder_nhead=8, 
                              eeg_encoder_dim_feedforward = 2048, embed_dim=768, model_path=model_path, llm_path=llm_path)


    state_dict = torch.load(checkpoint_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path, 
               score_results=score_results, input_type = test_input, 
               pretrained_model = pretrained, llama2Tokenizer=llama2Tokenizer, embedding_model=embedding_model)
