import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from transformers import BertTokenizer
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig

def init_tokenizer(truncation_side="right"):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side=truncation_side)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    return tokenizer

class llm4rec(nn.Module):
    def __init__(
        self,
        llm_model="",
        max_output_txt_len=256,
    ):
        super().__init__()
        self.tokenizer = init_tokenizer(truncation_side="left")
        
        if llm_model == 'opt':
            self.llm_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7b", use_fast=False)
            self.llm_model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16)
        else:
            raise Exception(f'{llm_model} is not supported')
            
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token
        self.llm_tokenizer.add_special_tokens({'additional_special_tokens': ['[UserRep]','[HistoryEmb]','[CandidateEmb]']})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]
        
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        self.max_output_txt_len = max_output_txt_len

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len

    def replace_hist_candi_token(self, llm_tokens, inputs_embeds, interact_embs, candidate_embs):
        if len(interact_embs) == 0:
            return llm_tokens, inputs_embeds
        history_token_id = self.llm_tokenizer("[HistoryEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        candidate_token_id = self.llm_tokenizer("[CandidateEmb]", return_tensors="pt", add_special_tokens=False).input_ids.item()
        
        for inx in range(len(llm_tokens["input_ids"])):
            idx_tensor=(llm_tokens["input_ids"][inx]==history_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, interact_embs[inx]):
                inputs_embeds[inx][idx]=item_emb
        
            idx_tensor=(llm_tokens["input_ids"][inx]==candidate_token_id).nonzero().view(-1)
            for idx, item_emb in zip(idx_tensor, candidate_embs[inx]):
                inputs_embeds[inx][idx]=item_emb
        return llm_tokens, inputs_embeds
    
    def forward(self, query_output, query_tokens, samples, mode):
        if mode == 'next_item_prediction_mlp':
            log_emb = query_output
            
            if log_emb != '':
                device = log_emb.device
                atts_llm = torch.ones(log_emb.size()[:-1], dtype=torch.long).to(log_emb.device)
                atts_llm = atts_llm.unsqueeze(1)
            else:
                device = samples['interact'][0].device
                
            text_output_tokens = self.llm_tokenizer(
                [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
                return_tensors="pt",
                padding="longest",
                truncation=False,
            ).to(device)
            
            text_input_tokens = self.llm_tokenizer(
                samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=False,
            ).to(device)
            
            llm_tokens, input_part_targets_len = self.concat_text_input_output(
                text_input_tokens.input_ids,
                text_input_tokens.attention_mask,
                text_output_tokens.input_ids,
                text_output_tokens.attention_mask,
            )

            targets = llm_tokens['input_ids'].masked_fill(
                llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
            )

            for i, l in enumerate(input_part_targets_len):
                targets[i][:l] = -100
            
            if log_emb != '':
                empty_targets = (
                    torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
                )

                targets = torch.cat([empty_targets, targets], dim=1)
            
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
            llm_tokens, inputs_embeds = self.replace_hist_candi_token(llm_tokens, inputs_embeds, samples['interact'], samples['candidate'])
            attention_mask = llm_tokens['attention_mask']
            
            if log_emb != '':
                log_emb = log_emb.unsqueeze(1)
                inputs_embeds = torch.cat([log_emb, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        
        with torch.cuda.amp.autocast():
            outputs = self.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
        loss = outputs.loss

        return loss