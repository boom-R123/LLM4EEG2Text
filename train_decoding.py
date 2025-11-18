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
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderConfig, EncoderDecoderModel
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive
from model import LLMTranslator
from config import get_config
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, BertModel
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import random



def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt', train_input='EEG', pretrained_model=None):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_loss = 100000000000
    num = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, \
                sentiment_labels, word_content, word_text_embeddings, word_token_nums, word_negative_embedding, subject, word_list in tqdm(dataloaders[phase]):
                # load in batch
                input_embeddings_batch = input_embeddings.float().to(device)
                word_text_embeddings = word_text_embeddings.float().to(device)
                word_negative_embedding = word_negative_embedding.float().to(device)
                word_token_nums = word_token_nums.to(device)
                word_tokens_ids = torch.zeros(input_embeddings.shape[0], input_embeddings.shape[1]).to(device).long()
                for l in range(len(word_list)):
                    words = word_list[l].split("<DIV>")
                    words_token = tokenizer(words, return_tensors='pt', padding='max_length', max_length=5, truncation=True, add_special_tokens=False)["input_ids"]
                    word_tokens_ids[l, :words_token.shape[0]] = words_token[:, 0]


                mask = (input_mask_invert==0)
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)

                """replace padding ids in target_ids with -100"""
                target_ids_batch[~target_mask.bool()] = -100
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, 
                                            word_text_embeddings, 
                                            word_negative_embedding,
                                            word_tokens_ids,
                                            epoch=epoch)
                    """calculate loss"""
                    # NOTE: my criterion not used
                    loss = seq2seqLMoutput.loss # use the BART language modeling loss
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.sum().backward()
                        # accelerator.backward(loss.sum())
                        optimizer.step()
                # statistics
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # batch loss
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent' #unique_sent unique_subj
    
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    train_input = args['train_input']
    dataset_path = args['dataset_path']
    model_path = args['model_path']
    llm_path = args['llm_path']


    print("train_input is:", train_input)   
    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    device_ids = [0] # device setting

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4
        
    print(f'[INFO]using model: {model_name}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    
    if use_random_init:
        save_name = 'randinit_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')


    ''' set random seeds '''
    seed_val = 888
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    random.seed(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()


    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = dataset_path +  'task1-SR/pickle/task1-SR-dataset.pickle'
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

    """save config"""
    cfg_dir = dataset_path + 'config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained(model_path)

    elif model_name == 'LLMTranslator':
        tokenizer = BertTokenizer.from_pretrained(model_path)

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, 
                             subject = subject_choice, eeg_type = eeg_type_choice, 
                             bands = bands_choice, setting = dataset_setting, test_input=train_input, model_path=model_path)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, 
                           subject = subject_choice, eeg_type = eeg_type_choice, 
                           bands = bands_choice, setting = dataset_setting, test_input=train_input, model_path=model_path)

    
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    # print('[INFO]test_set size: ', len(test_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = batch_size, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    if model_name == 'BrainTranslator':
        if use_random_init:
            config = BartConfig.from_pretrained(model_path)
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained(model_path)
    
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'BrainTranslatorNaive':
        pretrained = BartForConditionalGeneration.from_pretrained(model_path)
        model = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'LLMTranslator':
        pretrained = BertModel.from_pretrained(model_path).to(device)
        model = LLMTranslator(in_feature = 105*len(bands_choice), eeg_encoder_nhead=8, 
                              eeg_encoder_dim_feedforward = 2048, embed_dim = 768,
                              model_path=model_path, llm_path=llm_path)
    model.to(device)

    

    ''' training loop '''

    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False
 

    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)


    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        if "pretrained" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # for name, param in pretrained.named_parameters():
    #     param.requires_grad = False

    ''' set up optimizer and scheduler'''

    # optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    optimizer_step2 = optim.AdamW(model.parameters(), lr=step2_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)


    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()
    
    print()
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)
    
    '''main loop'''
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, train_input=train_input, pretrained_model=pretrained)

