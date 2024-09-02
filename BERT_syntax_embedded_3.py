# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:51:04 2023

@author: hast
"""

import os
import re
import urllib.request
import tarfile
import random
import glob
from tqdm import tqdm
import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MPNetModel
MODEL_NAME='microsoft/mpnet-base'
#cuda=True
cuda=False
#print_similarity
printline=False
#test mode for tokenize process
test=False
test1=0
test2=1
#if bind lines when tokenizing
bunch_of_lines=5
#switch_secondlines: if True, ignore {2: and treat as normal line
switch_secondlines=True
#switch_insertion: if True, ignore {i:, no insertion,i.e. original
# original
switch_insertion=True
# revised
switch_insertion=False
#For evaluation. Number of top similar pairs to evaluate
evaluatenum=15


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = MPNetModel.from_pretrained(MODEL_NAME)
if cuda:
    mode = model.cuda()
#max_length=768
max_length=256
if test:
    max_length=80
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

files='texts/*.txt'

basic_tokens='{ } : S V O C E 2 i '
elements_items=['S','V','O','C','E','S2','V2','O2','C2','E2','2','i']
if switch_secondlines:
    elements_items=['S','V','O','C','E','2','i']
elements_items_2=['{S:','{V:','{O:','{C:','{E:','{S2:','{V2:','{O2:','{C2:','{E2:','{2:']
CLS='CLS'
SEP='SEP'
INSERT='{i:'
poems=[]
text=''
tokens=[]
tokens_e=[]
spos=1
epos=1
shift=0
tokenized=[]
current_pos=0
max_pos=0
tokens2=0
num_braces=[]
start=0
end=0
dic_tokens={}
dic_tokens2=[]
ids=[]
current_type=''
dic_tokens={}
dic_embeddings={}
dic_embeddings_average={}
tokenized_table=[]
embeddings_table=[]
trans_embeddings_table=[]
transfer_table=[]
files=''
files_list=[]
titles=[]
last_hidden_states=[]
tokenized_table=[]
encoding=[]
encoding2=[]
outputs=[] 
attention_mask=[]
embedding_texts=[]
flag_last=False
end_of_sentence2=[]    

def get_basic_tokens_ids(basic_tokens=basic_tokens):
    encoding=tokenizer(basic_tokens,padding='max_length')
    basic_tokens_ids={}
    j=1
    for k in basic_tokens:
        if k!=' ':
            basic_tokens_ids[k]=encoding['input_ids'][j]
            j += 1
    basic_tokens_ids[CLS]=encoding['input_ids'][0]
    basic_tokens_ids[SEP]=encoding['input_ids'][j]
    return basic_tokens_ids    

def get_SVO_tokens_ids(basic_tokens_ids=get_basic_tokens_ids(),
                       elements_items=elements_items, CLS=CLS, SEP=SEP):
    SVO_tokens_ids={}
    for k in elements_items:
        SVO_tokens_ids[k]=[]
        for v in range(len(k)):
            SVO_tokens_ids[k]=[basic_tokens_ids['{'], basic_tokens_ids[k[0]]]
            if v==1:
                SVO_tokens_ids[k].append(basic_tokens_ids[k[1]])
                break
        SVO_tokens_ids[k].append(basic_tokens_ids[':'])
    SVO_tokens_ids['}']=  [basic_tokens_ids['}']]
    SVO_tokens_ids[CLS]=  [basic_tokens_ids[CLS]]
    SVO_tokens_ids[SEP]=  [basic_tokens_ids[SEP]]
    return SVO_tokens_ids

#key position, next brace position and next braseclose position
def get_start_end_pos(text, key, start=0, end=len(text)):
    key_pos=text.find(key, start, end)
    next_brace=text.find('{',key_pos+1,end)
    next_barceclose=text.find('}',key_pos+1,end)
    return key_pos, next_brace, next_braceclose

#relevant brace position from the start value in text
def get_relevant_brace(text, start=0, end=len(text)):
    num_brace=0
    relevant_pos=0
    key='{|}'
    brace_list=[m.span() for m in re.finditer(key, text)]
    brace_pos=[]
    print(start)
    print(brace_list)
    for k,v in brace_list:
        if k>=start:
            brace_pos.append([k,v])
            if text[k:v]=='{':
                num_brace+=1
                print('+1')
            if text[k:v]=='}':
                num_brace-=1
                print('-1')
            if num_brace==0:
                relevant_pos=k
                print('break', k)
                break
    if num_brace>0:
        if num_brace == 1:
            relevant_pos=end
        if num_brace>1:
            relevant_pos=brace_pos[1][0]
    print('relevant_pos:', relevant_pos)
    return relevant_pos
            
def tokenize_only(text, max_length=max_length):
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
        )
    return encoding

def bert_text(text, max_length=max_length, cuda=cuda):
    encoding = tokenize_only(text, max_length)
    if cuda:
        encoding = {k: v.cuda() for k,v in encoding.items()}
    attention_mask=encoding['attention_mask']
    with torch.no_grad():
        outputs = model(**encoding)
        last_hidden_states = outputs.last_hidden_state
    return last_hidden_states, encoding, outputs, attention_mask

def index_multi(list, x):
    return [m for m, _x in enumerate(list) if _x == x]

#get a next braceclose position in the tokens list
def next_bracesclose(tokens=tokens, start=spos, end=epos, key='}'):    
    num_bracesclose=index_multi(tokens, key)
    epos=end
    for m in num_bracesclose:
        if start<m:
            epos=m
            break
    return epos

#get a next brace or braceclose position in the tokens list
def next_braces_or_bracesclose(tokens=tokens, start=spos, end=epos):
    k=index_multi(tokens, '}')
    v=index_multi(tokens, '{')
    num_braces_or_bracesclose=k+v
    num_braces_or_bracesclose.sort()
    print('num_braces_or_bracesclose :',num_braces_or_bracesclose)
    epos=len(tokens)-1
    if len(num_braces_or_bracesclose)>0:
        for m in num_braces_or_bracesclose:
            #print('start',start)
            if start<m:
                epos=m
                print('start,m: ',start,m)
                break
    return epos

def next_relevant_braceclose(tokens=tokens, start=spos, end=epos):
    k=index_multi(tokens, '}')
    v=index_multi(tokens, '{')
    num_braces_or_bracesclose=k+v
    num_braces_or_bracesclose.sort()
    print('num_braces_or_bracesclose: ', num_braces_or_bracesclose)
    k=0
    v=len(tokens)
    if len(num_braces_or_bracesclose)>0:
        for m in num_braces_or_bracesclose:
            if start<=m:
               if tokens[m]=='{':
                   k=k+1
                   print('m: ', m,"+1")
               if tokens[m]=='}':
                   k=k-1
                   print('m: ', m,"-1")
                   
                   if k<=0:
                       v=m
                       break
    return v                   
  
#calculating shift, spos, epos    
def cal_shift(start=spos, end=epos, shift=shift, tokens=tokens):
    if tokens[start]=='{':
        if next_bracesclose(tokens, start, end, ':')-start==2:
            shift=shift+4            
        if next_bracesclose(tokens, start, end, ':')-start==3:
            shift=shift+5
    return shift

def check_pad(epos=epos,tokens=tokens):
    for v in range(len(tokens)):
        if tokens[epos]=='<pad>':
            epos=epos-1
        else:
            break
    return epos

#pop element from text: spos, epos, ids, not type
def pop_element(start=current_pos, end=max_pos, shift=shift, current_type=current_type,
                tokens=tokens, tokenized=tokenized, tokens2=tokens2,
                current_pos=current_pos,end_of_sentence2=end_of_sentence2):
    print('---pop_element---')
    print('tokens:',tokens)
#    print('tokenized:',tokenized)
    print('start:',start)
    print('end:',end)
    spos=start
    epos=end
    ids=[]
    type=current_type
    spos_embeddings=0
    epos_embeddings=0
    if start<end:
        if tokens[start]=='{':
#           epos=next_bracesclose(tokens, spos, epos)
            epos=next_braces_or_bracesclose(tokens, spos, epos)             
            type=tokens[spos+1:next_bracesclose(tokens, spos, epos, ':')][0]
            if tokens2==1:
                type=type+'2'
#check in get_element def
#            if type=='2':
#                type='E2'
            shift=cal_shift(spos, epos, shift, tokens)
            print('shift: ', shift)
            epos=check_pad(epos,tokens)
            ids=tokenized[0][next_bracesclose(tokens, spos, epos, ':')+1:epos]
            spos_embeddings=next_bracesclose(tokens, spos, epos, ':')-shift+2
            epos_embeddings=epos-shift+1
            
        else:
            epos=next_braces_or_bracesclose(tokens, spos, epos)
            epos=check_pad(epos,tokens)
            ids=tokenized[0][spos:epos]
            spos_embeddings=spos-shift
            epos_embeddings=epos-shift
            print('epos:', epos)
            if tokens[epos]=='}':
                shift+=1
                
    print('shift:',shift)
    print('current_type:',current_type)
    return spos, epos, ids, type, shift, spos_embeddings, epos_embeddings        
    
def push_dic_token(dic_tokens=dic_tokens, spos=spos, epos=epos, ids=ids, 
                   current_type=current_type, tokens=tokens):
    d2=tokenizer.convert_ids_to_tokens(ids)    
    d=[{'spos': spos, 'epos': epos, 'ids': ids, 'conv': d2}]
    current_type=current_type.upper()
    print(current_type)
    d2=dic_tokens[current_type]
    d2.append(d)
    dic_tokens[current_type]=d2
    return dic_tokens

def check_current_type(tokens2=tokens2, current_type=current_type, elements_items=elements_items):
    print('check_current_type: ',tokens2,current_type)
    current_type=current_type.upper()
    if not switch_secondlines:
        if tokens2==1:    
            for m in range(0,4):
                if current_type==elements_items[m]:
                    current_type=elements_items[m+5]
        else:
            for m in range(5,10):
                if current_type==elements_items[m]:
                    current_type=elements_items[m-5]
    return current_type

def check_last(tokens=tokens, current_pos=current_pos, max_length=max_length):
    print('current_pos: ', current_pos)
    print('tokens: ', tokens)
    flag_last=False
    if current_pos==max_length:
        flag_last=True
    else:
        if tokens[current_pos]=='<pad>':
            flag_last=True
    return flag_last

def check_sentence2(end_of_sentence2=end_of_sentence2, current_pos=current_pos):
    tokens2=0
    for k,v in end_of_sentence2:
        print('kv: ', k,v)
        if current_pos>=k and current_pos<=v:
            tokens2=1
            print('check_sentence2 True')
    return tokens2

def check_last_sentence2(end_of_sentence2=end_of_sentence2, current_pos=current_pos):
    tokens2=0
    for es,ee in end_of_sentence2:
        if current_pos==ee:
            tokens2=1
            print('check_last_sentence2 True')
    return tokens2
    
def get_elements(text=text, 
                 elements_items=elements_items, elements_items_2=elements_items_2,
                 CLS=CLS, SEP=SEP, INSERT=INSERT, 
                 switch_secondlines=switch_secondlines, switch_insertion=switch_insertion):

    #source_text
    source_text=text
    #original_text
    original_text=source_text.replace(INSERT,'').replace('}','')
    for i in elements_items_2:
        original_text=original_text.replace(i,'')
    #tokenize_text    
    tokenize_text=source_text
    if INSERT in tokenize_text:
        j=[m.span() for m in re.finditer(INSERT, tokenize_text)]
        for k in range(len(j)):
            i=tokenize_text.find(INSERT)
            relevant_pos=get_relevant_brace(tokenize_text, i)
            print('i,j,relevant_pos,k: ',i,j,relevant_pos,k)
            if tokenize_text[relevant_pos]=='}':
                if switch_insertion:
                    #{i:}
                    tokenize_text=tokenize_text[:i]+tokenize_text[relevant_pos+1:]
#                    k=k+(relevant_pos-i)
                    print('switch: i, relevant, k: ',i,relevant_pos, k)
                else:
                    #}
                    tokenize_text=tokenize_text[:relevant_pos]+' '+tokenize_text[relevant_pos+1:]
                    #{i:
                    tokenize_text=tokenize_text[:i]+tokenize_text[i+len(INSERT):]
#                    k=k+len(INSERT)
    #embedding_text
    embedding_text=tokenize_text.replace('}','')
    for i in elements_items_2:
        embedding_text=embedding_text.replace(i,'')
    #tokenizing: mkaing sure the position of the embedding_text
    encoding=tokenize_only(tokenize_text, max_length)
    tokenized=encoding['input_ids'].cpu().numpy()
    tokens=tokenizer.convert_ids_to_tokens(tokenized.tolist()[0])
    encoding_e=tokenize_only(embedding_text, max_length)
    tokenized_e=encoding_e['input_ids'].cpu().numpy()
    tokens_e=tokenizer.convert_ids_to_tokens(tokenized_e.tolist()[0])
    basic_type='E'
    tokens2=0
    #second clause will be treated differently
    num_braces=index_multi(tokens, '{')
    max_pos=len(tokens)
    #end of sentence2: braceclose pos for {2: sentence} 
    end_of_sentence2=[]
    for i in num_braces:
        if tokens[i+1]=='2' and tokens[i+2]==':':
            m=next_relevant_braceclose(tokens, i, max_pos)
            end_of_sentence2.append([i,m])

    current_pos=1
    spos=1
    epos=1
    shift=0
    current_type=basic_type
    spos_embeddings=0
    epos_embeddings=0
    dic_tokens={}
    dic_tokens2={}
    #dic_tokens
    for i in elements_items:
        dic_tokens[i]=[]
    if len(num_braces)>0:
        #left side before the first {
        if current_pos<index_multi(tokens, '{')[0]:
            print('---left side before the first {---')
            print('current_type')
            print(current_type)
            print(len(num_braces))
            print('num_braces:',num_braces)
            m=next_braces_or_bracesclose(tokens, spos, epos)
            print('m: ', m)
            tokens2=check_sentence2(end_of_sentence2, current_pos)
            flag_last=check_last(tokens,current_pos,max_length)
            if not flag_last:
                spos, epos, ids, current_type, shift, spos_embeddings, epos_embeddings=pop_element(current_pos, m,shift,current_type, tokens, tokenized)
                print('spos: ', spos)
                print('epos: ', epos)
                print('current_type: ', current_type)
                dic_tokens=push_dic_token(dic_tokens, spos_embeddings, epos_embeddings, ids, current_type)
                current_pos=m
                print('current_pos: ', current_pos) 

        #other parts {
        for i in num_braces:
            m=next_braces_or_bracesclose(tokens, current_pos)
            current_type=basic_type
            tokens2=check_sentence2(end_of_sentence2, current_pos)
            #for {2:
            if tokens2==1:
#            if tokens[i+1]=='2':
                basic_type='E2'
                if switch_secondlines:
                    basic_type='E'
                current_type=basic_type

            #for normal elements after {
            print('---other parts---')
            print('i: ', i)
            print('m: ', m)
            flag_last=check_last(tokens,current_pos,max_length)
            if not flag_last:
                spos, epos, ids, current_type, shift, spos_embeddings, epos_embeddings=pop_element(i, m,shift,current_type, tokens, tokenized)
                current_type=check_current_type(tokens2, current_type)
                print('spos: ', spos)
                print('epos: ', epos)
                print('current_type: ', current_type) 
                if current_type=='2':
                    current_pos=epos
                    shift-=1
                    current_type='E2'
                    if switch_secondlines:
                        current_type='E'
#                    print('current_type=2, current_pos=epos')
                else:
                    current_pos=epos+1
                print(ids)
                if len(ids)!=0:
                    dic_tokens=push_dic_token(dic_tokens, spos_embeddings, epos_embeddings, ids, current_type)
                print('***current_pos: ', current_pos)
                #} for {2:
                tokens2=check_last_sentence2(end_of_sentence2, current_pos)
                if tokens[current_pos]=='}' and tokens2==1:
                    current_pos+=1
                    shift+=1
            
            #after }
            m=next_braces_or_bracesclose(tokens, current_pos)
            current_type=basic_type
            tokens2=check_sentence2(end_of_sentence2, current_pos)
            #braceclose for {2:
            if tokens[current_pos-1]=='}' and check_last_sentence2(end_of_sentence2, current_pos-1)==1:
                basic_type='E'
                current_type=basic_type
            print('---other parts after }---')
            print('m: ', m)
            flag_last=check_last(tokens,current_pos,max_length)
            if not flag_last:
                if not tokens[current_pos]=='{':
                    spos, epos, ids, current_type, shift, spos_embeddings, epos_embeddings=pop_element(current_pos, m,shift,current_type, tokens, tokenized)
                    current_type=check_current_type(tokens2, current_type)
                    print('spos: ', spos)
                    print('epos: ', epos)
                    print('current_type: ', current_type) 
                    dic_tokens=push_dic_token(dic_tokens, spos_embeddings, epos_embeddings, ids, current_type)
#                    current_pos=epos+1
                    current_pos=epos+1
                    print('current_pos: ', current_pos)
                    #} for {2:
                    tokens2=check_last_sentence2(end_of_sentence2, current_pos)
                    if tokens[current_pos]=='}' and tokens2==1:
                        current_pos+=1
                        shift+=1
                        
        #last parts
        if current_pos>num_braces[-1]:
            print('---right side after the last }---')
            print('current_type')
            print(current_type)
            print(len(num_braces))
            print('num_braces:',num_braces)
            m=max_pos
            current_type='E'
            tokens2=check_sentence2(end_of_sentence2, current_pos)
            print('m: ', m)
            #pad
            flag_last=check_last(tokens,current_pos,max_length)
            if not flag_last:
                spos, epos, ids, current_type, shift, spos_embeddings, epos_embeddings=pop_element(current_pos, m,shift,current_type, tokens, tokenized)
                print('spos: ', spos)
                print('epos: ', epos)
                print('current_type: ', current_type)
                dic_tokens=push_dic_token(dic_tokens, spos_embeddings, epos_embeddings, ids, current_type)
                current_pos=m
                print('current_pos: ', current_pos)
    else:
        print('---no braces---')
        print('current_type')
        print(current_type)
        print(len(num_braces))
        print('num_braces:',num_braces)
        m=max_pos
        current_type='E'
        print('m: ', m)
        spos, epos, ids, current_type, shift, spos_embeddings, epos_embeddings=pop_element(current_pos, m,shift,current_type, tokens, tokenized)
        print('spos: ', spos)
        print('epos: ', epos)
        print('current_type: ', current_type)
        dic_tokens=push_dic_token(dic_tokens, spos_embeddings, epos_embeddings, ids, current_type)
        current_pos=m
        print('current_pos: ', current_pos)
    #dic_tokens2 for return value
    dic_tokens2={}
    dic_tokens2['source_text']=source_text
    dic_tokens2['original_text']=original_text
    dic_tokens2['tokenize_text']=tokenize_text
    dic_tokens2['embedding_text']=embedding_text
    dic_tokens2['tokens']=dic_tokens
    dic_tokens2['tokens_e']=tokens_e    
    return dic_tokens2

#read files: files='texts/*.txt'
def read_files(files='texts/*.txt'):
    document=[]
    files_list=[]
    for file in glob.glob(files):
        files_list.append(file)
        poem=open(file).read().split('\\n')
        poems=[]
        for lines in poem:
            line=lines.strip().replace('\\n','')
            line=re.sub('\s+',' ',line)
            poems.append(line)
        document.append(poems)
    return document, files_list

def print_tokenized_table(tokenized_table=tokenized_table, 
                          elements_items=elements_items):
    bar='-----------------------------------'
    CR='\n'
    line=[]
    for i in range(len(tokenized_table)):
#        print(bar)
        line.append(bar)
        if 'title' in tokenized_table[i]:
#            print(tokenized_table[i]['title'])
            line.append(tokenized_table[i]['title'])
#        print(bar)
        line.append(bar)
        if not 'tokens_table' in tokenized_table[i]:
            break
        for j in range(len(tokenized_table[i]['tokens_table'])):
#            print(CR)
            line.append(CR)
#            print('document and line:', i,j)
            line.append('document and line:'+' '+str(i)+' '+str(j))
            if 'tokens_table' in tokenized_table[i] and 'dict' in tokenized_table[i]['tokens_table'][j]:
                dic=tokenized_table[i]['tokens_table'][j]['dict']
            else: 
#                print('Blank')
                line.append('Blank')
                break
#            print(dic['tokenize_text'])
            line.append(dic['tokenize_text'])
            L=''
            i2=dic['tokens_e']
            for j2 in i2:
                if j2!='<pad>':
                    L=L+j2+' '
#            print('tokens: '+L)
            line.append('tokens_e: '+L)
            #print(CR)
            for k in elements_items:
                #print(k)
                for v in range(len(dic['tokens'][k])):
                    #print(dic['tokens'][k])
                    #print(len(dic['tokens'][k]))
                    s=dic['tokens'][k][v][0]
                    L=' '.join(s['conv'])
#                    print(k+' '+'spos: '+str(s['spos'])+\
#                          ' epos:'+str(s['epos'])+' conv:'+L)
                    line.append(k+' '+'spos: '+str(s['spos'])+\
                          ' epos:'+str(s['epos'])+' conv:'+L)

    if printline:
        for i in line:
            print(i)
    return line            

def create_tokenized_table(poems=poems, bunch_of_lines=bunch_of_lines, titles=titles,
                           elements_items=elements_items, 
                           elements_items_2=elements_items_2,
                           CLS=CLS, SEP=SEP, INSERT=INSERT):
    #preprocessing for gathering lines
    #\p
    para_sign='\p'
    poems0=poems
    poems2=[]
    for document_num, document in enumerate(poems0):
        text=[]
        for k in document:
            #paragraph split
            if para_sign in k:
                v=k.find(para_sign)
                text.append(k[:v+len(para_sign)])
                text.append(k[v+len(para_sign):])
            else:
                if k!='':
                    text.append(k)
        #a bunch of lines
        texts=[]
        line=''
        m=0
        for k in text:
            if para_sign in k:
                k.replace(para_sign,'')
                line=line+k+' '
                m=bunch_of_lines
            else:
                line=line+k+' '
                m+=1
            if m>=bunch_of_lines:
                texts.append(line)
                line=''
                m=0
        poems2.append(texts)
    poems=poems2        
        
    
    tokenized_table=[{}]
    embedding_texts=[]
    for i in range(len(poems)-1):
        tokenized_table.append([{}])
    for document_num, document in enumerate(poems):
        tokenized_table[document_num]={'document_num': document_num}
        if 'title' in tokenized_table[document_num]:
            tokenized_table[document_num]={'title': titles[document_num]}
        tokenized_table[document_num]['tokens_table']=[[{}]]
        print(document)
        for i in range(len(document)-1):
            tokenized_table[document_num]['tokens_table'].append([[{}]])
        i=0
        if test==True and document_num>test1:
            break
        for line in document:
            print('line:', line)
            if test==True and i > test2:
                break
            dic_tokens2=get_elements(line,elements_items, elements_items_2,CLS, SEP, INSERT)
            if dic_tokens2['embedding_text']!='' or dic_tokens2['tokens_e'][2]!='<pad>':
                dic_tokens2['ll']=i
                tokenized_table[document_num]['tokens_table'][i]={'ll': i}            
                tokenized_table[document_num]['tokens_table'][i]={'dict': dic_tokens2}
                i+=1
                embedding_text=dic_tokens2['embedding_text']
                embedding_texts.append(embedding_text)
            else:
                #erase blank line
                x=tokenized_table[document_num]['tokens_table'].pop(i)
        #print(tokenized_table)
        line=print_tokenized_table(tokenized_table)
        with open('tokenized_table.txt','w') as f:
            f.writelines(([d+"\n" for d in line]))
    return tokenized_table, embedding_texts

def create_transfer_table(last_hidden_states=last_hidden_states, tokenized_table=tokenized_table,
                          encoding=encoding, outputs=outputs, 
                          attention_mask=attention_mask, embedding_texts=embedding_texts,
                          elements_items=elements_items,CLS=CLS,SEP=SEP, max_length=max_length,
                          cuda=cuda, switch_secondlines=switch_secondlines):
    hiddensize=model.config.hidden_size
#    hiddensize=max_length
#    transfer_table=torch.Tensor(len(last_hidden_states),hiddensize*10)
    transfer_table=[]
    refdata=last_hidden_states
    num_lines=[]
    num_lines2=[]
    num_documents=len(tokenized_table)
    num_gelements=10
    if switch_secondlines:
        num_gelements=5
#    #concatenated transferred vector
#    transferred_vector=torch.Tensor(hiddensize*10,len(last_hidden_states))
    #temporary vector
    vec=torch.zeros(hiddensize)
    vec2=torch.zeros(hiddensize*num_gelements)
    vec_zero=torch.zeros(hiddensize)
    ll=0
    num_SVO_tokens=0
    #each document(work)
    #作品
    for i in range(num_documents):
       num_lines.append(len(tokenized_table[i]['tokens_table']))
       #each line
       #詩行
       for j in range(len(tokenized_table[i]['tokens_table'])):
           dic=tokenized_table[i]['tokens_table'][j]['dict']
           line=tokenized_table[i]['tokens_table']
           if 'dict' in line:
               if 'tokens' in line['dict']:
                   tokendata=dic['tokens']
           #each item ({'S': [], 'V': [], 'O': [], 'C': [], 'E': [[{'spos': 1, 'epos': 3, 'ids':...)
           pos=0
           vec2=torch.zeros(hiddensize*num_gelements)
           #({'S': [], 'V': [], 'O': [], 'C': [], 'E':
           #文法項目ごと
           for k in elements_items[0:num_gelements]:
               num_SVO_tokens=0
               #print('k: ',k)
               if dic['tokens'][k]=='':
                   vec=vec_zero
                   num_SVO_tokens=1
                   num_lines2.append('no element')
               else:
                   #同じ文法項目内の行ごと
                   for v in range(len(dic['tokens'][k])):
                       #print('v: ',v)
                       spos=dic['tokens'][k][v][0]['spos']
                       epos=dic['tokens'][k][v][0]['epos']
                       x='i,j,k,v,spos,epos: '+str(i)+' '+str(j)+' '+str(k)+' '+str(v)+' '+str(spos)+' '+str(epos)
                       num_lines2.append(x)
                       #ids: multiple word hidden layers in one item: ku ##bla khan
                       #同じ行のトークンごと
                       for n in range(0,epos-spos):
                           #print('n: ',n)
                           x=str(refdata[ll][spos+n][0])
                           num_lines2.append(str(n)+' '+x+' spos: '+str(spos)+' n: '+str(n)+' ll: '+str(ll))
                           vec=vec+refdata[ll][spos+n]
                               #print(vec)
                               #print(len(vec))
                           num_SVO_tokens+=1
               #average
               if num_SVO_tokens==0:
                   vec=vec_zero
               else:
                   vec=vec/num_SVO_tokens
               x='average: '+str(+vec[0])
               num_lines2.append(x)
                   #
               vec2[pos*hiddensize:(pos+1)*hiddensize]=vec
               x='pos*hiddensize: '+str(pos*hiddensize)
               num_lines2.append(x)
               vec=torch.zeros(hiddensize)
               pos+=1
#           transferred_vector[:,j]=vec2
           ll+=1
           vec2=torch.nan_to_num(vec2)
           if cuda:
               transfer_table.append(vec2.cpu().numpy())
           else:
               transfer_table.append(vec2.numpy())
        
    return transfer_table       
                       
                   

    
print('loading files...')
poems, files_list=read_files('texts/*.txt')
titles=files_list
for i in range(len(titles)):
    titles[i]=titles[i].replace('texts\\','')
    titles[i]=titles[i].replace('.txt','')
    

print('tokenizing...')
tokenized_table, embedding_texts=create_tokenized_table(poems, bunch_of_lines, titles)
#tokenized_table, embedding_texts=create_tokenized_table(poems, bunch_of_lines, elements_items, 
#                                                        elements_items_2, CLS, SEP, INSERT)

print('BERT calculating...')
starttime=time.time() 
last_hidden_states, encoding, outputs, attention_mask=bert_text(embedding_texts,max_length)
endtime=time.time()
print('BERT time:', endtime-starttime)
last_hidden_states0=last_hidden_states

print('transferring...')
transfer_table=create_transfer_table(last_hidden_states, tokenized_table,
                          encoding, outputs, 
                          attention_mask, embedding_texts,
                          elements_items,CLS,SEP, max_length,cuda)



sentence_vectors=np.vstack(transfer_table)

#cos_sim by sentence vectors
def cos_sim(sentence_vectors=sentence_vectors, tokenized_table=tokenized_table):
    norm=np.linalg.norm(sentence_vectors, axis=1, keepdims=True)
    sentence_vectors_normalized=sentence_vectors / norm
    sim_matrix=sentence_vectors_normalized.dot(sentence_vectors_normalized.T)
    return sim_matrix

sim_matrix=cos_sim(sentence_vectors, tokenized_table)

#cos sim necessary parts
def cos_sim_small(sim_matrix=sim_matrix, tokenized_table=tokenized_table):
    num_poem1=len(tokenized_table[0]['tokens_table'])
    num_poem2=len(tokenized_table[1]['tokens_table'])
    if num_poem1+num_poem2==len(sentence_vectors):
        sim_matrix_2by2=sim_matrix[0:num_poem1, num_poem1:num_poem1+num_poem2]
    else:
        sim_matrix_2by2=False
    return sim_matrix_2by2

#最終的なコサイン類似度　KK x PL
sim_matrix_2by2=cos_sim_small(sim_matrix, tokenized_table)

#描画最後

#select top cos_sims
def cos_sim_top(sim_matrix_2by2=sim_matrix_2by2,printnum=30):
    top=[(i,j,sim_matrix_2by2[i,j]) for i in range(sim_matrix_2by2.shape[0]) for j in range(sim_matrix_2by2.shape[1])]
    top.sort(key=lambda x: x[2])
    top.reverse()

    printline=[]
    printline.append('-----------------------')
    printline.append('A='+titles[0]+'   B='+titles[1])
    printline.append('-----------------------')
    for k in range(printnum):
        i=top[k][0]
        j=top[k][1]
        value=top[k][2]
        printline.append('A: '+tokenized_table[0]['tokens_table'][i]['dict']['embedding_text'])
        printline.append('B: '+tokenized_table[1]['tokens_table'][j]['dict']['embedding_text'])
        printline.append('value: '+str(value))
        printline.append('-----------------------')
    return top, printline

printnum=evaluatenum
top, printline=cos_sim_top(sim_matrix_2by2,printnum)
#print('\n------\nPBP\n------\n')
#for i in printline:
#    print(i)    

#normal BERT
averaged_hidden_state=\
    (last_hidden_states*attention_mask.unsqueeze(-1)).sum(1)\
        /attention_mask.sum(1, keepdim=True)
svector=averaged_hidden_state.cpu().numpy()
#svector=np.vstack(svector)

sim_matrix_sBERT=cos_sim(svector, tokenized_table)
sim_matrix_2by2_sBERT=cos_sim_small(sim_matrix_sBERT, tokenized_table)
top_sBERT, printline_sBERT=cos_sim_top(sim_matrix_2by2_sBERT,printnum)
#print('\n------\nNormal BERT\n------\n')
#for i in printline_sBERT:
#    print(i)      

#Evaluation
#top 5 similar texts compared to human evaluation
# number of coinsidence with human evaluation / number of similar pairs
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
#!pip install Levenshtein
import Levenshtein

dir='C:\\書庫\\研究計画2023\\DH2024\\EMNLP\\'
data_file_Kubla_Khan='Kubla_Khan_embeddings_categorized.csv'
data_file_Paradise_Lost='Paradise_Lost_embeddings_categorized.csv'

KKPL_name=["line","text","category"]
KK=pd.read_csv(dir+data_file_Kubla_Khan, header=0, names=KKPL_name)
PL = pd.read_csv(dir+data_file_Paradise_Lost, header=0, names=KKPL_name)

#テキストごと、行数ごとにテキストをまとめるpdをつくる
def add_bunch_of_lines(df=KK):
    df['text']=df['text'].str.strip()
    loc=df.columns.get_loc('text')
    loc2=df.columns.get_loc('category')
    index_max=df.shape[0]
    df2 = df
    if bunch_of_lines != 1 and bunch_of_lines != 0:
        for i in range(index_max):
            for j in range(bunch_of_lines):
                if i+j < index_max and j != 0:
                    df2.iat[i,loc]=df2.iat[i,loc]+' '+df2.iat[i+j,loc]
                    if not pd.isnull(df2.iat[i,loc2]):
                        if not pd.isnull(df2.iat[i+j,loc2]):
                            df2.iat[i,loc2]=df2.iat[i,loc2]+','+df2.iat[i+j,loc2]
                    else:
                        if not pd.isnull(df2.iat[i+j,loc2]):
                            df2.iat[i,loc2]=df2.iat[i+j,loc2]
            if not pd.isnull(df2.iat[i,loc2]):
                df2.iat[i,loc2]=list(set(df2.iat[i,loc2].split(',')))
                #print(df2.iat[i,loc2])
    return df2

def add_bunch_of_lines_0(df=KK):
    df['text']=df['text'].str.strip()
    loc=df.columns.get_loc('text')
    loc2=df.columns.get_loc('category')
    index_max=df.shape[0]
    df2 = df
    if bunch_of_lines != 1 and bunch_of_lines != 0:
        for i in range(index_max):
            for j in range(bunch_of_lines):
                if i+j+1 < index_max:
                    df2.iat[i,loc]=df2.iat[i,loc]+' '+df2.iat[i+j+1,loc]
                    if not pd.isnull(df2.iat[i,loc2]):
                        if not pd.isnull(df2.iat[i+j+1,loc2]):
                            df2.iat[i,loc2]=df2.iat[i,loc2]+','+df2.iat[i+j+1,loc2]
                    else:
                        if not pd.isnull(df2.iat[i+j+1,loc2]):
                            df2.iat[i,loc2]=df2.iat[i+j+1,loc2]
            if not pd.isnull(df2.iat[i,loc2]):
                df2.iat[i,loc2]=list(set(df2.iat[i,loc2].split(',')))
                #print(df2.iat[i,loc2])
    return df2

KK2=add_bunch_of_lines(KK)
PL2=add_bunch_of_lines(PL)

#テキストごとに人間の基準を取得
df3=KK2
def get_eval_category(df3=KK2,text_num=0,top0=top):
#    df3 = df3
#    text_num = text_num
    eval = []
    for k in range(evaluatenum):
        #PBP top
    #top_normalBERT
        i=top0[k][text_num]
        texti=tokenized_table[text_num]['tokens_table'][i]['dict']['embedding_text']
        max_ratio, max_text, max_ratio_num, m = 0, None, 0, 0
        for j in df3['text'].to_list():
            ratio=Levenshtein.ratio(texti,j)
            if ratio > max_ratio:
                max_ratio, max_text, max_ratio_num = ratio, j, m
            m += 1
#        n = df3.iat[max_ratio_num, df3.columns.get_loc('category')]
#        if pd.isnull(df3.iat[max_ratio_num, df3.columns.get_loc('category')]):
#            n=[]
        eval.append([k,max_ratio,texti, max_text, max_ratio_num, df3.iat[max_ratio_num, df3.columns.get_loc('category')],i])
    #print(eval)
    l=pd.DataFrame(eval)
    l[5]=l[5].fillna('['']')
    eval=l.values.tolist()
    #print(eval)
    return eval

#最後の5の要素にカテゴリーのリストが入る
KK3=get_eval_category(KK2,0,top)
PL3=get_eval_category(PL2,1,top)

a=[]
def remove_chara(chara=",", a=a):
    if chara in a:
        a.remove(chara)
    return a
        
#これを二つのテキストで共通する要素を抽出し1つ以上あれば1とカウントして、
#evaluatenumを分母として割合を出す。
def get_eval_accuracy(L1=KK3, L2=PL3):
    eval_count,eval_accuracy = 0,0
    L1_L2_and = []
    for i in range(evaluatenum):
#        if pd.isnull(L1[i][5]).all():
#            L1[i][5]=['']
#        if pd.isnull(L2[i][5]).all():
#            L2[i][5]=['']
        KK_PL_and = list(set(L1[i][5]) & set(L2[i][5]))
#        print('top '+str(i+1)+': ', KK_PL_and)
        KK_PL_and=remove_chara('[',KK_PL_and)
        KK_PL_and=remove_chara(']',KK_PL_and)
        KK_PL_and=remove_chara(',',KK_PL_and)
        L1_L2_and.append(KK_PL_and)
        if len(KK_PL_and)!=0:
            eval_count+=1
    eval_accuracy = eval_count / evaluatenum
    return eval_accuracy, L1_L2_and

L1_L2_and=0
def get_printline2(top=top, printnum=15, L1=KK3, L2=PL3, shared_topics=L1_L2_and):

    printline=[]
    printline.append('-----------------------')
    printline.append('A='+titles[0]+'   B='+titles[1])
    printline.append('-----------------------')
    for k in range(printnum):
        value=top[k][2]
        printline.append('A: '+str(L1[k][2]))
        printline.append('topic: '+str(L1[k][5])+'  number: '+str(L1[k][6]))
        printline.append('B: '+L2[k][2])
        printline.append('topic: '+str(L2[k][5])+'  number: '+str(L2[k][6]))
        printline.append('value: '+str(value))
        printline.append('shared topics: '+str(shared_topics[k]))
        printline.append('-----------------------')
#    for i in printline:
#        print(i)
    return printline


def get_original_insertion(original_insertion='original'):
    original_insertion=original_insertion
    if switch_insertion!=True:
        original_insertion='insertion(revised)'
    return original_insertion

original_insertion=get_original_insertion()
eval_accuracy, L1_L2_and = get_eval_accuracy(KK3,PL3)
outfile=[]

outfile.append('\n------\nPBP\n------\n')
printline=get_printline2(top, evaluatenum, KK3, PL3, L1_L2_and)
for i in printline:
    outfile.append(i)
title_PBP=str(bunch_of_lines)+' line(s): PBP: '+original_insertion
outfile.append(title_PBP)
outfile.append('Evaluation: '+str(eval_accuracy))
outfile.append('-----------------------')


#normal BERT
#最後の5の要素にカテゴリーのリストが入る
KK3_sBERT=get_eval_category(KK2,0,top_sBERT)
PL3_sBERT=get_eval_category(PL2,1,top_sBERT)
eval_accuracy_sBERT, L1_L2_and_sBERT = get_eval_accuracy(KK3_sBERT,PL3_sBERT)

outfile.append('\n------\nNormal BERT\n------\n')
printline=get_printline2(top_sBERT, evaluatenum, KK3_sBERT, PL3_sBERT, L1_L2_and_sBERT)
for i in printline:
    outfile.append(i)
title_sBERT=str(bunch_of_lines)+' line(s): sBERT: '+original_insertion
outfile.append(title_sBERT)
outfile.append('Evaluation: '+str(eval_accuracy_sBERT))
outfile.append('-----------------------')

for i in outfile:
    print(i)
outfile2=""
for i in outfile:
    outfile2=outfile2+i+"\n"
path_out=dir+'results\\'+'line_'+str(bunch_of_lines)+'_'+original_insertion+'_'+str(evaluatenum)+'.txt'
print('file: '+path_out)
with open(path_out, "w", encoding="utf-8") as f:
    f.write(outfile2)

#描画
def get_png(sim_temp=sim_matrix_2by2, title_temp=title_PBP, vmin=0.89):
    fig, ax = plt.subplots()
    sns.heatmap(sim_temp, ax=ax, cmap="YlGnBu", vmin=vmin)
    ax.set_ylabel(titles[0])
    ax.set_xlabel(titles[1])
    ax.set_title(title_temp)
    title_temp2=re.sub(r':','', title_temp)
    title_temp2=re.sub(r' ','_', title_temp2)
    plt.savefig(dir+'results\\'+title_temp2+'_'+str(evaluatenum)+'.png')           
    plt.show()

top_15_cs_vmin=[[0,0],[0.89,0.95],[0.913,0.963],[0.92,0.969],[],[0.93,0.97]]
#top_15_cs_vmin=[[0,0],[0.89,0.95],[0.91503686,0.9643397],[0.92,0.969],[],[0.93,0.97]]

#PBP heatmap
get_png(sim_matrix_2by2, title_PBP, top_15_cs_vmin[bunch_of_lines][0])
#sBERT heatmap
get_png(sim_matrix_2by2_sBERT, title_sBERT, top_15_cs_vmin[bunch_of_lines][1])

#ideal heatmap
KK2_num=int(len(KK2)/bunch_of_lines)
PL2_num=int(len(PL2)/bunch_of_lines)
ideal=np.empty((KK2_num, PL2_num))
KK2['category']=KK2['category'].fillna('['']')
PL2['category']=PL2['category'].fillna('['']')
a=[]

for i in range(KK2_num):
    for j in range(PL2_num):
        a = list(set(KK2.iat[i*bunch_of_lines,2]) & set(PL2.iat[j*bunch_of_lines,2]))
        a=remove_chara('[',a)
        a=remove_chara(']',a)
        a=remove_chara(',',a)
        if len(a)>0:
            ideal[i][j]=1
        else:
            ideal[i][j]=0
get_png(ideal, 'ideal matrix(line(s) '+str(bunch_of_lines)+')', 1)

