#!/usr/bin/python
#coding=utf-8

import sys
import codecs
import re
import os
import time
import json
import docx

import jieba
import pandas

import jsonpickle
import math

INPUT_DIR_NAME = sys.argv[1]
DOC_TYPE = "sentence"

def make_document_multi_round(data_file):
    excel = pandas.ExcelFile(data_file)
    global id
    pairs = []
    for i in range(0, len(excel.sheet_names)):
        name = excel.sheet_names[i]
        if name == '分类' or name == 'List' or name == '闲聊' or name.startswith("_"):
            continue
        
        sheet = excel.parse(excel.sheet_names[i])
        values = sheet.values

        for j in range(0, values.shape[0]):
            question = values[j][0]
            answer = values[j][1]
            if type(question) == str and type(answer) == str:
                if not re.match("^[\\s]*$", question) and len(question) >= 2 and not re.match("^[\\s]*$", answer) and len(answer) >= 2:
                    pairs.append((question, answer))

    for (question, answer) in pairs:
        doc = {
            '_op_type': 'index',
            #'_index': INDEX_NAME,
            '_type': DOC_TYPE,
            '_id': id,
            '_source': {'question': question, 'answer': answer}
        }
        id = id + 1
        yield (doc)

def build_index(pairs):
    global total_docs
    total_docs = 0
    global inverted_index
    inverted_index = {}
    global forward_index
    forward_index = {}

    for pair in pairs:
        forward_index[pair['_id']] = pair['_source']['question'] + '    ' + pair['_source']['answer']

        #count global document counts
        total_docs += 1

        #split Chinese words
        terms = jieba.cut(pair['_source']['question'], cut_all=True)
        terms = [term for term in terms if not re.match("^[\\s]*$", term)]

        tf = {}
        for term in terms:
            if term in tf:
                tf[term]+=1
            else:
                tf[term] = 1

        for key, value in tf.items():
            entry = PostingEntry()
            entry.doc_id = pair['_id']
            entry.tf = value
            entry.doc_length = len(tf)

            #print(entry.to_string())

            if not key in inverted_index:
                post = Posting()
                post.word = key
                post.df = 0

                inverted_index[key] = post

            inverted_index[key].list.append(entry)
            inverted_index[key].df += 1

    for key, value in inverted_index.items():
        df = inverted_index[key].df
        inverted_index[key].idf = math.log(total_docs/df, 2)

    #output forward index to forward_index.json
    with open('forward_index.json', 'w') as outfile1:
        json.dump(forward_index, outfile1)

    #output inverted index to inverted_index.json
    frozen = jsonpickle.encode(inverted_index)
    with open('inverted_index.json', 'w') as outfile2:
        json.dump(frozen, outfile2)


class Posting:
    word = ''
    df = 0
    idf = 0
    list = []

    def __init__(self):
        self.word = ''
        self.df = 0
        self.idf = 0
        self.list = []

    def to_string(self):
        rslt = self.word + ':' + str(self.df) + ' ' + str(self.idf) + ' '
        for item in self.list:
             rslt += item.to_string()
        return rslt


class PostingEntry:
    doc_id = ''
    tf = 0
    doc_length = 0

    def __init__(self):
        self.doc_id = ''
        self.tf = 0
        self.doc_length = 0

    def to_string(self):
        return '(' + str(self.doc_id) + ':' + str(self.tf) +  ':' + str(self.doc_length) + ')'

#call simple_index_builder: python simple_index_builder.py corpus_path
global id 
id = 1

if __name__ == "__main__":
    files = os.listdir(sys.argv[1])
    id = 0 # start document id from 0

    for file in files:
        if "xlsx" in file:
            print("processing " + file)
            path = os.path.join(sys.argv[1], file)
            qa_pairs = make_document_multi_round(path)

            build_index(qa_pairs)
