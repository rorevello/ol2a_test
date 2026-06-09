#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 10:09:00 2025

@author: maciek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 17:30:35 2023

@author: ryb003
"""

import faiss
import numpy as np
from faiss import write_index, read_index
# write_index(index, "large.index")
# index = read_index("large.index")
import openai
import time
import pickle

class FaissManager:
    def __init__(self ):
        print ('Faiss starting.')
        self.index=None
        self.vectoriser=None
    
    def create_index(self, xb, ids):
        print(xb.shape)
        self.index = faiss.IndexFlatL2(xb.shape[1])
        self.index = faiss.IndexIDMap2(self.index)
        self.index.add_with_ids(xb, ids) # works, the vectors are stored in the underlying index
    
    def create_index_from_strings(self, texts, ids):
        xb = self.vectoriser.vectorise(texts)
        self.create_index(xb, ids)
    
    def add_to_index(self, xb, ids, no_duplicates=False):
        if no_duplicates==False:
            self.index.add_with_ids(xb, ids) # works, the vectors are stored in the underlying index
        else:
            all_ids=faiss.vector_to_array(self.index.id_map)
            all_ids=set(all_ids)
            for i, v in zip(ids, xb):
                if i not in all_ids:
                    self.index.add_with_ids(np.array([v]), np.array([i]))
                else:
                    print ('Dropping '+ str(i))
    
    def add_texts_to_index(self, texts, ids, no_duplicates=False):
        xb = self.vectoriser.vectorise(texts)
        self.add_to_index(xb, ids, no_duplicates)
        
    def vector_search(self, v, k=100):
        n=min(k, self.index.ntotal)
        distances, ann=self.index.search(np.array([v]), n)
        return distances, ann
    
    def text_search(self, text, prompt='Find relevant documents', k=100):
        v=self.vectoriser.vectorise([text], prompt)
        return self.vector_search(v[0], k)
    
    def rescore(self, ids, v):
        index = faiss.IndexFlatL2(v.shape[-1])
        index = faiss.IndexIDMap2(index)
        xb=[]
        for i in ids:
            # print(i)
            xb.append(self.index.reconstruct(int(i)))
        xb=np.array(xb)
        index.add_with_ids(xb, ids)
        distances, ann=index.search(np.array([v]), index.ntotal)
        return distances, ann
    
    def rescore_text(self, ids, q):
        v=self.vectoriser.vectorise([q])[0]
        return self.rescore(ids, v)
    
    def save_index(self, out):
        write_index(self.index, out)
        with open(out+'.vect', 'wb') as f:
            pickle.dump(self.vectoriser,f) 

    
    def load_index(self, idx, reload_vectoriser=False):
        self.index=read_index(idx)
        if reload_vectoriser:
            with open(idx+'.vect', 'rb') as f:
                self.vectoriser = pickle.load(f)


from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
import torch

class JinaVectoriser:
    def __init__(self, model="jinaai/jina-embeddings-v4", trust_remote=True, left_side_padding=False):
        self.count=0
        quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                        # llm_int4_threshold=6.0 
                    )
        
        if left_side_padding:
            tokenizer_conf={"padding_side": "left"}
        else:
            tokenizer_conf={}
        self.model = SentenceTransformer(
            model, trust_remote_code=trust_remote,
            model_kwargs={#"attn_implementation": "flash_attention_2",
                          "quantization_config": quantization_config, "device_map": "auto" },
            tokenizer_kwargs=tokenizer_conf,
        )
        
       
    def vectorise(self, texts, prompt='', batch=1):
        
        result=[]
        for i in range(0, len(texts), batch):
            if i != 0:
                result=np.concatenate((result,
                                       self.get_vector(texts[i:i+batch],
                                                       prompt=prompt)), axis=0)
                # result.extend(self.get_vector(texts[i:i+batch], prompt=prompt))
            else:
                result=self.get_vector(texts[i:i+batch], prompt=prompt)
        return np.array(result)

    def get_vector(self, text, prompt='', task="text-matching"):
        
        # print (text)
        if len(prompt)>0:
            embeddings = self.model.encode([prompt + ' ' + t for t in text], task=task)
        else:
            embeddings = self.model.encode(text, task=task)
        self.count+=len(text)
        # print(self.count)
        
        return embeddings
        
class QwenVectoriser:
    def __init__(self, model="Qwen/Qwen3-Embedding-8B", trust_remote=True, 
                 left_side_padding=True, instruct_tag=False):
        self.count=0
        quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                        # llm_int4_threshold=6.0 
                    )
        
        if left_side_padding:
            tokenizer_conf={"padding_side": "left"}
        else:
            tokenizer_conf={}
        self.model = SentenceTransformer(
            model, trust_remote_code=trust_remote,
            model_kwargs={#"attn_implementation": "flash_attention_2",
                          "quantization_config": quantization_config, "device_map": "auto" },
            processor_kwargs=tokenizer_conf,
        )
        self.instr=instruct_tag
        
       
    def vectorise(self, texts, prompt='', batch=1):
        
        result=[]
        for i in range(0, len(texts), batch):
            print(i)
            if i != 0:
                result=np.concatenate((result,
                                       self.get_vector(texts[i:i+batch],
                                                       prompt=prompt)), axis=0)
                # result.extend(self.get_vector(texts[i:i+batch], prompt=prompt))
            else:
                result=self.get_vector(texts[i:i+batch], prompt=prompt)
        return np.array(result)

    def get_vector(self, text, prompt=''):
        
        # print (text)
        if len(prompt)>0:
            if self.instr:
                embeddings = self.model.encode(text, prompt="Instruct: "+prompt+"\nQuery: ")
            else:
                embeddings = self.model.encode(text, prompt=prompt)
        else:
            embeddings = self.model.encode(text)
        self.count+=len(text)
        # print(self.count)
        
        return embeddings
        
class NVVectoriser:
    def __init__(self, model="nvidia/NV-Embed-v2", trust_remote=True):
        self.count=0
        
        self.model = SentenceTransformer(
            model, trust_remote_code=trust_remote)
            # model_kwargs={#"attn_implementation": "flash_attention_2",
            #               "quantization_config": quantization_config, "device_map": "auto" },
        # )
            
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side="right"
           
    def vectorise(self, texts, prompt='', batch=4):
            
        result=[]
        for i in range(0, len(texts), batch):
            if i != 0:
                result=np.concatenate((result,
                                       self.get_vector(texts[i:i+batch],
                                                       prompt=prompt)), axis=0)
                    # result.extend(self.get_vector(texts[i:i+batch], prompt=prompt))
            else:
                result=self.get_vector(texts[i:i+batch], prompt=prompt)
        return np.array(result)
        
    def add_eos(self, input_examples):
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples
        
    def get_vector(self, text, prompt=''):
            
        # print (text)
        inputs=self.add_eos(text)
        if len(prompt)>0:
            inp=["Instruct: "+prompt+"\nQuery: " + t for t in inputs]
            embeddings = self.model.encode(inp, batch_size=len(inputs), prompt=prompt, normalize_embeddings=True)
        else:
            embeddings = self.model.encode(inputs, batch_size=len(inputs), normalize_embeddings=True)
        self.count+=len(text)
        # print(self.count)
            
        return embeddings

if __name__=='__main__':
    
    # Usage example with vectors
    # m=FaissManager()
    
    # xb=[[0,0],
    #           [1,1],
    #           [1,2],
    #           [3,1],
    #           [2,1],
    #           [0,5]]
    # xb=np.array(xb)
    # ids = np.array([11, 22, 33, 44, 55, 66])
    # m.create_index(xb, ids)
    # m.save_index('test.index')
    # m.add_to_index(xb[:2, :], np.array([1, 22]), no_duplicates=True)
    # print(faiss.vector_to_array(m.index.id_map))
    # dist, nn=m.vector_search([0,0])
    # d,n=m.rescore(np.array([11,1]), np.array([1,1]))
    # m.load_index('test.index')
    # print(faiss.vector_to_array(m.index.id_map))
    # m.add_to_index(xb[:2, :], np.array([1, 22]), no_duplicates=False)
    # print(faiss.vector_to_array(m.index.id_map))
    
    
    # Usage example with strings
    m=FaissManager()
    m.vectoriser=QwenVectoriser()
    m.create_index_from_strings(['Michael Jordan is the best NBA player', 'Porridge is the healthiest food', 'Dry fractionation separates protein bodies from starch and fats.'], [23,0,1])
    di, ne = m.text_search('basketball')
    print(di)
    print(ne)
    # m.save_index('test_qwen.index')
    # m=FaissManager()
    # m.load_index('test_qwen.index')
    # di2, ne2 = m.text_search('food production')
    
