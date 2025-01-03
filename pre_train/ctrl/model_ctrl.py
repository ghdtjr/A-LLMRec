import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import pickle
import random
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class SASRec_CTRL(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec_CTRL, self).__init__()

        self.kwargs = {'user_num': user_num, 'item_num':item_num, 'args':args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.description = args.use_description

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        #Modality encoder
        self.sbert = SentenceTransformer('nq-distilbert-base-v1')
        
        #W_text in equation 3
        self.projection = torch.nn.Linear(768,args.hidden_units)
        
        #W_tab in equation 2
        self.projection2 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        
        # Fine-grained align Wm in equation 7
        self.finegrain1_1 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        self.finegrain1_2 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        self.finegrain1_3 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        self.finegrain2_1 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        self.finegrain2_2 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        self.finegrain2_3 = torch.nn.Linear(args.hidden_units,args.hidden_units)
        
        self.final_layer = torch.nn.Linear(args.hidden_units,args.hidden_units)
        
        #Backbone network
        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.args =args
        self.bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()

        #Load textual meta data
        with open(f'./data/Movies_and_TV_meta.json.gz','rb') as ft:
            self.text_name_dict = pickle.load(ft)
        
        #Backbone network
        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    # Find Text Data
    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"Title:{self.text_name_dict[t].get(i,t_)}, Description:{self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"Title:{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"Description:{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, 
                                            attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default', pretrain=True, opt = None):       
        
        # Get Collaborative model embedding (i.e., get Tab embedding in CTRL)
        log_feats = self.log2feats(log_seqs)

        # Cross-modal contrastive learning
        if pretrain:
            total_loss = 0
            iterss = 0
            log_feats = log_feats.reshape(-1,log_feats.shape[2])
            log_feats = log_feats[log_seqs.reshape(log_seqs.size)>0]
            text_list = []
            
            # Get Textual Data
            for l in log_seqs:
                ll = l[l>0]
                for i in range(len(ll)):
                    to_text = ll[:i+1]
                    text = "This is a a user, who has recently watched  " + '|'.join(self.find_item_text(to_text, description_flag=False))
                    text += '. This is a movie, title is ' + ','.join(self.find_item_text(to_text, description_flag=self.description))
                    print(text)
                    text_list.append(text)
            
            # Embed textual data using Semantic Model
            token = self.sbert.tokenize(text_list)
            text_embedding= self.sbert({'input_ids':token['input_ids'].to(log_feats.device),'attention_mask':token['attention_mask'].to(log_feats.device)})['sentence_embedding']
            
            # Projection - Equation 2 and 3
            text_embedding = self.projection(text_embedding)
            log_feats = self.projection2(log_feats)
            
            start_idx = 0
            end_idx = 32
            loss = 0
            
            # Cross-modal Contrasive Learning (Batch samples, auto-regressive - SASRec)
            while start_idx <len(text_embedding):
                
                cal = 0
                log = log_feats[start_idx:end_idx]
                text_ = text_embedding[start_idx:end_idx]
                start_idx+=32
                end_idx +=32
                iterss +=1
                
                # Fine-grained alignment
                # m-th sub-representation (m=3) -> h^tab_m (i.e., use output of collaborative model)
                log_fine1 = self.finegrain1_1(log)
                log_fine2 = self.finegrain1_2(log)
                log_fine3 = self.finegrain1_3(log)
                
                # m-th sub-representation (m=3) -> h^text_m (i.e., use output of semantic model)
                text_fine1 = self.finegrain2_1(text_)
                text_fine2 = self.finegrain2_2(text_)
                text_fine3 = self.finegrain2_3(text_)
                
                sim_mat1 = torch.matmul(log_fine1, text_fine1.T).unsqueeze(0)
                sim_mat2 = torch.matmul(log_fine1, text_fine2.T).unsqueeze(0)
                sim_mat3 = torch.matmul(log_fine1, text_fine3.T).unsqueeze(0)
                
                # Maximum similarity
                results1 = torch.cat([sim_mat1,sim_mat2,sim_mat3],dim=0).max(axis=0)[0]

                sim_mat4 = torch.matmul(log_fine2, text_fine1.T).unsqueeze(0)
                sim_mat5 = torch.matmul(log_fine2, text_fine2.T).unsqueeze(0)
                sim_mat6 = torch.matmul(log_fine2, text_fine3.T).unsqueeze(0)
                
                # Maximum similarity
                results2 = torch.cat([sim_mat4,sim_mat5,sim_mat6],dim=0).max(axis=0)[0]

                
                
                sim_mat7 = torch.matmul(log_fine3, text_fine1.T).unsqueeze(0)
                sim_mat8 = torch.matmul(log_fine3, text_fine2.T).unsqueeze(0)
                sim_mat9 = torch.matmul(log_fine3, text_fine3.T).unsqueeze(0)
                
                # Maximum similarity
                results3 = torch.cat([sim_mat7,sim_mat8,sim_mat9],dim=0).max(axis=0)[0]
                
                # Get fine-grained similarity over all sub-representations
                results = results1 + results2 + results3
                
                # Maximum similarity
                # Get fine-grained similarity over all sub-representations
                test_results1 = torch.cat([sim_mat1,sim_mat4,sim_mat7],dim=0).max(axis=0)[0]
                test_results2 = torch.cat([sim_mat2,sim_mat5,sim_mat8],dim=0).max(axis=0)[0]
                test_results3 = torch.cat([sim_mat3,sim_mat6,sim_mat9],dim=0).max(axis=0)[0]
                test_results = test_results1 + test_results2 + test_results3
                
                
                # tabular2textual Equation 5
                pos_labels = torch.ones(results.diag().shape, device=log_feats.device)
                neg_labels = torch.zeros(results[~torch.eye(len(text_),dtype=bool)].shape, device=log_feats.device)
                
                cal += self.bce_criterion(results.diag(), pos_labels)
                cal+=self.bce_criterion(results[~torch.eye(len(results),dtype=bool)], neg_labels)
                
                # textual2tabular Equation 4
                pos_labels = torch.ones(test_results.diag().shape, device=log_feats.device)
                neg_labels = torch.zeros(test_results[~torch.eye(len(text_),dtype=bool)].shape, device=log_feats.device)
                
                cal += self.bce_criterion(test_results.diag(), pos_labels)
                cal+=self.bce_criterion(test_results[~torch.eye(len(test_results),dtype=bool)], neg_labels)
                
                # Equation 6
                loss += (cal/2)
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            return total_loss
        
        else:
            # Supervised Fine-tuning
            log_feats = self.final_layer(log_feats)
            if mode == 'log_only':
                log_feats = log_feats[:, -1, :]
                return log_feats
                
            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

            if self.args.pretrain_stage == True:
                return log_feats.reshape(-1,log_feats.shape[2]), pos_embs.reshape(-1,log_feats.shape[2]), neg_embs.reshape(-1,log_feats.shape[2])
            else:
                return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        log_feats = self.final_layer(log_feats)
        
        final_feat = log_feats[:, -1, :]

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)


        return logits 
