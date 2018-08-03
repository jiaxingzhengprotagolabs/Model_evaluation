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

INDEX_NAME = sys.argv[1]
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
            '_index': INDEX_NAME,
            '_type': DOC_TYPE,
            '_id': id,
            '_source': {'question': question, 'answer': answer}
        }
        id = id + 1
        yield (doc)


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
        return '(' + str(self.doc_id) + ':' + str(self.tf) + ')'

class QueryTerm:
    term = ''
    qtf = 0
    idf = 0

    def __init__(self):
        self.term = ''
        self.qtf = 0
        self.idf = 0

    def to_string(self):
        return '(' + str(self.term) + ':' + str(self.qtf) + ':' + str(self.idf) + ')'

class ScoredDoc:
    doc_id = 0
    score = 0
    confident = 0

    def __init__(self):
        self.doc_id = ''
        self.score = 0
        self.confident = 0

    def to_string(self):
        return str(self.doc_id) + ' ' + str(self.score)


def retrieval(query):
    terms = jieba.cut(query, cut_all=True)
    terms = [term for term in terms if not re.match("^[\\s]*$", term)]

    qtf = {}
    for term in terms:
        if term in qtf:
            qtf[term] += 1
        else:
            qtf[term] = 1

    qterms = []
    for key, value in qtf.items():
        qterm = QueryTerm()
        qterm.term = key
        qterm.qtf = value

        qterm.idf = 0
        if key in inverted_index:
            qterm.idf = inverted_index[key].idf
        qterms.append(qterm)

    qterms.sort(key=lambda x: x.idf, reverse=True)

    print("Query is:")
    msg = ""
    for term in qterms:
        msg = msg + ' ' + term.term
    print(msg)

    #return retrieveBasedOnIDF(qterms)
    return retrieveIDFDocLengthNorm(qterms)



def retrieveBasedOnIDF(qterms):
    docs = {}
    max_score = 0

    for qterm in qterms:
        if qterm.term in inverted_index:
            post = inverted_index[qterm.term]

            max_score += post.idf

            for doc in post.list:
                doc_id = doc.doc_id
                if not doc_id in docs:
                    doc_obj = ScoredDoc()
                    doc_obj.doc_id = doc_id
                    doc_obj.score = 0
                    doc_obj.confident = 0
                    docs[doc_id] = doc_obj

                docs[doc_id].score += qterm.idf
                docs[doc_id].confident += qterm.idf

    rslt = list(docs.values())

    rslt.sort(key=lambda x: x.score, reverse=True)

    if max_score > 0:
        for doc in rslt:
            doc.confident /= max_score

    return rslt

def retrieveIDFDocLengthNorm(qterms):
    docs = {}
    max_score = 0

    for qterm in qterms:
        if qterm.term in inverted_index:
            post = inverted_index[qterm.term]

            max_score += post.idf

            for doc in post.list:
                doc_id = doc.doc_id

                # This approach is similar to cosine similarity
                # One vector is the query vector, the value for each dimension is the term's idf value
                #   e.g. for query "hello world", its vector is (3.4, 4.5),
                #   where 3.4 is the idf value of "hello", and 4.5 is the idf of "world"
                #       * note: the values 3.4 and 3.5 are faked
                # Another vector is the doc vector, the value for each dimension is "1"
                #   e.g. for document "hello california, nice to say hello to you again!", its vector is (1,1,1,1,1,1,1)
                #   since we do not consider tf here. This is saying we only count unique terms
                # Note that original cosine similarity should normalize both vectors' length, however,
                # here since query length is constant, I removed the query normalization out of the calculation.

                doc_len_discount = math.sqrt(doc.doc_length)

                if not doc_id in docs:
                    doc_obj = ScoredDoc()
                    doc_obj.doc_id = doc_id
                    doc_obj.score = 0
                    doc_obj.confident = 0
                    docs[doc_id] = doc_obj

                if doc_len_discount > 0:
                    docs[doc_id].score += (qterm.idf / doc_len_discount)
                else:
                    docs[doc_id].score += qterm.idf
                docs[doc_id].confident += qterm.idf

    rslt = list(docs.values())

    rslt.sort(key=lambda x: x.score, reverse=True)

    if max_score > 0:
        for doc in rslt:
            doc.confident /= max_score

    return rslt

id = 1

if __name__ == "__main__":
    #read inverted index and forward index
    index1 = json.loads(open('inverted_index.json').read())
    global inverted_index 
    inverted_index = jsonpickle.decode(index1)

    global forward_index 
    with open('forward_index.json', 'r') as fp:
        forward_index = json.load(fp)

    #retrieval for a query
    query = sys.argv[1]

    rslt = retrieval(query)

    count = 0
    for line in rslt:
        count += 1
        if count <= 5:
            print(line.to_string() + '  ' + forward_index[str(line.doc_id)] + ' ' + str(line.confident))
