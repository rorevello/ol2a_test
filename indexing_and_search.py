#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:30:17 2026

@author: maciek
"""

import ir_datasets
import json

dataset = ir_datasets.load("cord19/fulltext/trec-covid")

docs_in_qrels=[]
for qrel in dataset.qrels_iter():
    qrel # namedtuple<query_id, doc_id, relevance, iteration>
    docs_in_qrels.append(qrel.doc_id)

docs_in_qrels = set(docs_in_qrels)

i=0
docs_for_indexing=[]
doc_id_index={}
for doc in dataset.docs_iter():
    doc_id=doc[0]
    if doc_id in docs_in_qrels and doc_id not in doc_id_index.values():
        doi=doc[2]
        date = doc[3]
        title=doc[1]
        abstract = doc[4]
        
        f_old=''
        content=title + '\n' + abstract + '\n'
        for f in doc[5]:
            if f.title!=f_old:
                # print(f.title)
                content += f.title
                content+= '\n'
            f_old=f.title
            content+=f.text
            content+= '\n'
        docs_for_indexing.append(content)
        doc_id_index[len(docs_for_indexing)-1]=doc_id

from persistent_index import FaissManager, QwenVectoriser
m=FaissManager()
m.vectoriser=QwenVectoriser(model="Qwen/Qwen3-Embedding-0.6B")
# Not indexing anymore, so this stays commented out:
# m.create_index_from_strings(docs_for_indexing, list(range(len(docs_for_indexing))))
# m.save_index('qwen.index')
m.load_index('qwen.index')
di2, ne2 = m.text_search('how do coronaviruses spread between bats', 
                         prompt='Find scientific documents that answer the question', k=100)
top_doc_index=ne2[0][0] # second zero is the the index on the ranked list
top_doc_id=doc_id_index[top_doc_index]
top_doc_content=docs_for_indexing[top_doc_index]