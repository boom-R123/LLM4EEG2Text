import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, LlamaTokenizer
import math
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer, BertTokenizer
import random
import time
import heapq



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # self.value_embedding = nn.Linear(c_in, d_model)
        self.value_embedding = nn.Sequential(
            nn.Linear(c_in, d_model * 2, bias=False),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.GELU(),
        )
        # self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.position_embedding = nn.Embedding(60, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        position_ids = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
        x = self.value_embedding(x) + self.position_embedding(position_ids)  # + self.position_embedding(x)
        return self.dropout(x)


class NAROutput():
    def __init__(self, loss, pred_p):
        self.loss = loss
        self.pred_p = pred_p



class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.3):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features1: torch.Tensor, features2: torch.Tensor, features3: torch.Tensor, features4: torch.Tensor) -> torch.Tensor:
        """
        Args:
            similarity_matrix: Tensor of shape (batch_size, batch_size)
                               The similarity matrix containing pairwise similarities
        """
        # Scale the similarity matrix by the temperature
        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        features3 = F.normalize(features3, dim=-1)
        features4 = F.normalize(features4, dim=-1)
        similarity_matrix_11 = features1 @ features1.T
        similarity_matrix_12 = features1 @ features2.T
        similarity_matrix_13 = features1 @ features3.T
        similarity_matrix_14 = features1 @ features4.T
        # similarity_matrix_42 = features4 @ features2.T
        # similarity_matrix_43 = features4 @ features3.T

        # similarity_matrix3 = similarity_matrix_11 / self.temperature
        # batch_size = similarity_matrix3.size(0)
        # similarity_matrix_22 = features2 @ features2.T
        # mask = (similarity_matrix_22) > 0.8
        # loss = -(similarity_matrix3[0][mask[0]].exp().sum() / similarity_matrix3[0].exp().sum()).log()
        # for i in range(1, batch_size):
        #     loss += -(similarity_matrix3[i][mask[i]].exp().sum() / similarity_matrix3[i].exp().sum()).log() 
        # print(similarity_matrix_11.mean(), similarity_matrix_11[mask].mean(), mask.float().sum() / batch_size)
        # return loss / batch_size + 1 - similarity_matrix_11[mask].mean()



        print(similarity_matrix_11.mean().item(), similarity_matrix_11.diag().mean().item())
        print(similarity_matrix_12.mean().item(), similarity_matrix_12.diag().mean().item())
        print(similarity_matrix_13.mean().item())
        print(similarity_matrix_14.mean().item(), similarity_matrix_14.diag().mean().item())

        # similarity_matrix = torch.cat([similarity_matrix_12, similarity_matrix_13], dim=1) / self.temperature
        # batch_size = similarity_matrix.size(0)
        # labels = torch.arange(batch_size).to(similarity_matrix.device)
        # loss = F.cross_entropy(similarity_matrix, labels)
        # return loss


        similarity_matrix1 = torch.cat([similarity_matrix_12.diag().unsqueeze(1), similarity_matrix_13], dim=1)
        similarity_matrix1 = similarity_matrix1 / self.temperature
        batch_size = similarity_matrix1.size(0)
        labels1 = torch.zeros(batch_size).long().to(similarity_matrix1.device)
        loss1 = F.cross_entropy(similarity_matrix1, labels1)

        similarity_matrix2 = similarity_matrix_12 / self.temperature
        batch_size = similarity_matrix2.size(0)
        labels2 = torch.arange(batch_size).to(similarity_matrix_12.device)
        loss2 = F.cross_entropy(similarity_matrix2, labels2)

        similarity_matrix3 = similarity_matrix_11 / self.temperature
        batch_size = similarity_matrix3.size(0)
        labels3 = torch.arange(batch_size).to(similarity_matrix3.device)
        loss3 = F.cross_entropy(similarity_matrix3, labels3)

        similarity_matrix4 = similarity_matrix_14 / self.temperature
        batch_size = similarity_matrix4.size(0)
        labels4 = torch.arange(batch_size).to(similarity_matrix4.device)
        loss4 = F.cross_entropy(similarity_matrix4, labels4)

        return (1 - similarity_matrix_12.diag().mean()) * 4 + similarity_matrix_13.mean() * 2 + loss4
        # triplet_loss = nn.TripletMarginLoss(margin=0.3, p=2)
        # return 0.5 * loss4 + 5 * triplet_loss(features1, features2, features3)
        # return loss2





class LLMTranslator(nn.Module):
    def __init__(self, pretrained_model=None, in_feature = 840, eeg_encoder_nhead=8, 
                 eeg_encoder_dim_feedforward = 1024, embed_dim = 512, model_path=None, llm_path=None):
        super(LLMTranslator, self).__init__()

        self.encoder_embedding = DataEmbedding(c_in=in_feature, d_model=embed_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=eeg_encoder_nhead,  dropout=0.3,
                                                        dim_feedforward = eeg_encoder_dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.project_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim*2, bias=False),
                                        nn.GELU(),
                                        nn.Dropout(0.3),
                                        nn.Linear(embed_dim*2, embed_dim, bias=False))
        self.Llama2tokenizer = LlamaTokenizer.from_pretrained(llm_path)
        self.Llama2tokenizer.pad_token_id = self.Llama2tokenizer.eos_token_id
        self.Berttokenizer = BertTokenizer.from_pretrained(model_path)


    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted, word_negative_embedding, word_tokens_ids, epoch=0):
        mask = (input_masks_invert==0)

        encoder_embed = self.encoder_embedding(input_embeddings_batch)
        encoder_out = self.encoder(encoder_embed, src_key_padding_mask=input_masks_invert.bool())
        encoder_out = self.project_layer(encoder_out)


        encoder_out = F.normalize(encoder_out, p=2, dim=-1)
        value, pred = (encoder_out[mask] @ encoder_out[mask].T - torch.eye(encoder_out[mask].size(0)).to(encoder_out.device)).softmax(1).max(dim=1)

        pred = word_tokens_ids[mask][pred]
        
        input_embeddings_batch_clone = input_embeddings_batch.clone()
        random_mask = (torch.rand_like(input_embeddings_batch_clone) < 0.3).bool()
        input_embeddings_batch_clone[random_mask] = torch.randn_like(input_embeddings_batch_clone[random_mask]).to(input_embeddings_batch_clone.device)
        encoder_embed_noise = self.encoder_embedding(input_embeddings_batch_clone)
        encoder_out_noise = self.encoder(encoder_embed_noise, src_key_padding_mask=input_masks_invert.bool())
        encoder_out_noise = self.project_layer(encoder_out_noise)
        criterion = InfoNCELoss()
        loss = criterion(encoder_out[mask], target_ids_batch_converted[mask], word_negative_embedding[mask], encoder_out_noise[mask])

        return NAROutput(loss=loss, pred_p=None)



    def forward_encode(self, input_embeddings_batch, input_mask_invert):
        encoder_embed = self.encoder_embedding(input_embeddings_batch)
        encoder_out = self.encoder(encoder_embed, src_key_padding_mask=input_mask_invert.bool())
        encoder_out = self.project_layer(encoder_out)
        return encoder_out
    

    @torch.no_grad()
    def generate(self, input_embeddings_batch, input_mask_invert, input_ids, word_token_nums, word_text_embeddings, LLM, embedding_model, max_length):
        encoder_embed = self.encoder_embedding(input_embeddings_batch)
        encoder_out = self.encoder(encoder_embed, src_key_padding_mask=input_mask_invert.bool())
        encoder_out = self.project_layer(encoder_out)
        num = (input_mask_invert.bool()==0).sum().item()
        num_return_sequences = 300 # 返回的候选结果数量
        for i in range(1, num):  
            with torch.no_grad():
                try:
                    # 实现beam search,获取每一步的beam search结果
                    s0, pred = LLM(input_ids).logits[:, -1, :].softmax(-1).topk(num_return_sequences, dim=-1)
                    outputs_1 = torch.cat((input_ids.repeat(num_return_sequences, 1), pred.transpose(0, 1)), dim=-1)
                    s1, pred = LLM(outputs_1).logits[:, -1, :].softmax(-1).topk(num_return_sequences, dim=-1)
                    s1 = (s0 * s1).reshape(-1)
                    outputs_2 = torch.cat((outputs_1.repeat_interleave(num_return_sequences, 0), pred.reshape(-1).unsqueeze(1)), dim=-1)
                    s1, idx = s1.topk(num_return_sequences, dim=-1)
                    outputs_2 = outputs_2[idx]
                    s2, pred = LLM(outputs_2).logits[:, -1, :].softmax(-1).topk(num_return_sequences, dim=-1)
                    s2 = (s1 * s2).reshape(-1)
                    outputs_3 = torch.cat((outputs_2.repeat_interleave(num_return_sequences, 0), pred.reshape(-1).unsqueeze(1)), dim=-1)
                    s2, idx = s2.topk(num_return_sequences, dim=-1)
                    outputs_3 = outputs_3[idx]
                    s3, pred = LLM(outputs_3).logits[:, -1, :].softmax(-1).topk(num_return_sequences, dim=-1)
                    s3 = (s2 * s3).reshape(-1)
                    outputs_4 = torch.cat((outputs_3.repeat_interleave(num_return_sequences, 0), pred.reshape(-1).unsqueeze(1)), dim=-1)
                    s3, idx = s3.topk(num_return_sequences, dim=-1)
                    outputs_4 = outputs_4[idx]
                except:
                    return input_ids
                outputs = torch.zeros_like(outputs_4).repeat(4, 1).cuda()
                outputs[:] = self.Llama2tokenizer.pad_token_id
                outputs[:outputs_1.shape[0], -outputs_1.shape[1]:] = outputs_1
                outputs[outputs_1.shape[0]:outputs_1.shape[0]*2, -outputs_2.shape[1]:] = outputs_2
                outputs[outputs_1.shape[0]*2:outputs_1.shape[0]*3, -outputs_3.shape[1]:] = outputs_3
                outputs[outputs_1.shape[0]*3:outputs_1.shape[0]*4, -outputs_4.shape[1]:] = outputs_4

            outputs_strings = self.Llama2tokenizer.batch_decode(outputs[:, :], skip_special_tokens=True)
            Bert_input = self.Berttokenizer(outputs_strings, return_tensors='pt', padding=True)
            Bert_input["input_ids"] = Bert_input["input_ids"].cuda()
            Bert_input["attention_mask"] = Bert_input["attention_mask"].cuda()
            with torch.no_grad():
                bert_output = embedding_model(input_ids=Bert_input["input_ids"], attention_mask=Bert_input["attention_mask"])['last_hidden_state']
            candi = []
            for b in range(bert_output.shape[0]):
                idx = Bert_input["attention_mask"][b].int().sum()-1
                candi.append(bert_output[b, idx-1, :])
            candi = torch.stack(candi)
            cos_sim = candi @ encoder_out[0, i, :] / (candi.norm(dim=-1) * encoder_out[0, i, :].norm())
            # cos_sim = candi @ word_text_embeddings[0, i, :] / (candi.norm(dim=-1) * word_text_embeddings[0, i, :].norm())
            input_ids = outputs[cos_sim.argmax()]
            input_ids = input_ids[input_ids!=self.Llama2tokenizer.eos_token_id].unsqueeze(0).cuda()
        return input_ids