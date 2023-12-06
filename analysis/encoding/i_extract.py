import copy
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPVisionModel, CLIPProcessor
from PIL import Image
from transformers import AutoImageProcessor

import sys

class TransformerRSM(object):

    def __init__(self, model_name, file_path=None, verbose=False):

        self.model_name = "openai/clip-vit-base-patch32"
        #self.model_name = "openai/clip-vit-large-patch14" ## this is the larger model. Gotta update torch for it tho
        
        self.verbose = verbose
        self.stimulus_df = pd.read_csv('./shrek_transcript_trs.csv')
        self.d_pathes = './shrek_tr_imgs/'

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Models can return full list of hidden-states & attentions weights at each layer
        ## Full model
        self.transformer = AutoModel.from_pretrained(self.model_name,
                                                     output_hidden_states=True,
                                                     output_attentions=True)
        
        ## model broken down by visual and text input stream
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.transformer_txt = CLIPTextModel.from_pretrained(self.model_name)
        self.transformer_vis = CLIPVisionModel.from_pretrained(self.model_name)
        
        # Modify the max_length attribute in the processor
        config = self.transformer_txt.config  # Use the text model's config (as an example)
        #new_max_tokens = 512  # Change this to your desired max tokens
        #config.max_position_embeddings = new_max_tokens
        
        # Load models with the updated configuration
        #transformer_txt = CLIPTextModel.from_pretrained(self.model_name, config=config)
        #transformer_vis = CLIPVisionModel.from_pretrained(self.model_name, config=config)
        #processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", clip_model=transformer_txt)
        
        #print(config)

        #self.transformer_txt.max_length = new_max_tokens
        #self.transformer_vis.max_length = new_max_tokens
        
        
        self.transformer.eval()

    # BASIC processing: generate embeddings and attention outputs for stimulus.
    def process_stimulus_activations(self, num_context_trs=5, save_activations=True, save_z_reps=True):
        
        self.stimulus_df.sentence.fillna('', inplace=True) # fill with empties
        trial_chunked_tokens = self.stimulus_df.sentence.values # convert to array
        raw_imgs = self.stimulus_df.stimulus.values
        num_h_layers = 13
     

        t_feats = []
        i_feats = []
        
        # Enumerate over the trial-aligned tokens
        for i, trial in enumerate(zip(raw_imgs, trial_chunked_tokens)):
            img = trial[0]
            txt = trial[1]
            if self.verbose and i % 10 == 0:
                print("Processing trial {}.".format(i))
                
            if len(txt) > 0: # if not empty, go to work
                # Get the full context window for this TR, e.g. the appropriate preceding number of TRs.
                context_window_index_start = max(5, i - num_context_trs) ## first 5 trs should not be included in context
                #print('context:', trial_chunked_tokens[context_window_index_start:i + 1])
                window_stimulus = " ".join(trial_chunked_tokens[context_window_index_start:i + 1])
                
                ### Encode Text ### 
                # Get the full context window for this TR, e.g. the appropriate preceding number of TRs.
                trial_token_ids = torch.tensor([
                    self.tokenizer.encode(window_stimulus) #self.tokenizer.encode(txt)
                ])

                ## Encode image ## 
                pth = self.d_pathes + img + '.png'
                image = Image.open(pth)
                print(np.array(image).shape)

                proc_d = self.processor(text=None, images=image, return_tensors="pt")

                outputs_txt = self.transformer_txt(trial_token_ids, output_hidden_states = True, output_attentions= True)
                outputs_vis = self.transformer_vis(proc_d['pixel_values'], output_hidden_states = True)

                h_layers_txt = outputs_txt.hidden_states
                h_layers_vis = outputs_vis.hidden_states
                attns = outputs_txt.attentions
                print('VIS hidden layer shape:', h_layers_vis[3].shape, 'TXT hidden layer shape:', h_layers_txt[3].shape)
                print('num hidden layers:', len(h_layers_vis))
                t_feats.append(h_layers_txt)
                i_feats.append(h_layers_vis)
            else: # if empty, skip word embedding but still grab visual embedding
                ## Encode image ## 
                pth = self.d_pathes + img + '.png'
                image = Image.open(pth)

                proc_d = self.processor(text=None, images=image, return_tensors="pt")

                outputs_vis = self.transformer_vis(proc_d['pixel_values'], output_hidden_states = True)

                h_layers_vis = outputs_vis.hidden_states
                h_layers_txt =  None #tuple([torch.zeros(1, 28, 512) for i in range(num_h_layers)])
                
                print('VIS hidden layer shape:', h_layers_vis[3].shape, 'NO TXT')
                print('num hidden layers:', len(h_layers_vis))
                t_feats.append(h_layers_txt)
                i_feats.append(h_layers_vis)

            if self.verbose:
                print("\t TR stimulus: {}".format(trial))

        ## Add new variables to dataframe
        self.stimulus_df["i_feats"] = i_feats
        self.stimulus_df["t_feats"] = t_feats
        
        ## forward fill -- add previous TRs values to all empty values
        self.stimulus_df["i_feats"].ffill(inplace=True)  
        self.stimulus_df["t_feats"].ffill(inplace=True)
        
        #self.stimulus_df["transformer_tokens_in_trial"] = trial #*update within loop if we want this info
        self.stimulus_df["n_transformer_tokens_in_tr"] = list(map(lambda x: len(x) if x else 0, trial_chunked_tokens))
        print("Processed {} TRs for activations.".format(len(i_feats)))

if __name__ == "__main__" :
    clip = TransformerRSM(model_name="openai/clip-vit-base-patch32", verbose=True)
    clip.process_stimulus_activations(save_activations=True, save_z_reps=True)
    clip.stimulus_df.to_csv('features.csv')
