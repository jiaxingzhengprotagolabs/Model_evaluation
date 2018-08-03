#!/usr/bin/python
#coding=utf-8

import numpy as np
import pandas as pd
import codecs
import sys
import re
import os
import time
import docx
import json
import jieba
import math
import jsonpickle

##sys.path.append("./")


## implementation of retrieval

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

	##print("Query is:")
	msg = ""
	for term in qterms:
		msg = msg + ' ' + term.term
	##print(msg)

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


### end of implementation of retrieval










def query_generate(file):
	## import test query
	test_question = pd.read_csv(file)
	## 提取指定列
	title_question = test_question.columns.values.tolist()[0]


	query = []
	for i in range(len(test_question)):
		query.append(test_question[title_question][i])

	return query



def ground_truth(f1, f2, f3):
	ground_truth1 = pd.read_csv(f1)
	ground_truth2 = pd.read_csv(f2)
	ground_truth3 = pd.read_csv(f3)
	ground_truth = [ground_truth1, ground_truth2, ground_truth3]
	title_ground_truth = ground_truth1.columns.values.tolist()[0]

	return ground_truth, title_ground_truth





def evaluation_data_gen(query, ground_truth, title):
	evl_data_set = []
	global inverted_index 
	global forward_index
	index1 = json.loads(open('inverted_index.json').read())
	inverted_index = jsonpickle.decode(index1)

	##global forward_index 
	with open('forward_index.json', 'r') as fp:
		forward_index = json.load(fp)

	
	##retrieve from corpus based on the test questions
	res_tmp = []
	for i in range(len(query)):
		rslt = retrieval(query[i])
		res_tmp.append(rslt)

	## generate the data set that can be used for evaluation
	for i in range(len(res_tmp)):
		count = 0
		model_res = []
		## 暂时设定只有前三句话 top前三
		for j in range(3):
					# id & socre       # Q & A -> Q                                         # confident score   #labe
			rank = [res_tmp[i][j].to_string(), forward_index[str(res_tmp[i][j].doc_id)].split('   ')[0].strip(), str(res_tmp[i][j].confident), 0]
			#print(line.to_string() + ' ' + forward_index[str(line.doc_id)] + ' ' + str(line.confident) )
			##print(forward_index[str(line.doc_id)])
			model_res.append(rank)
		evl_data_set.append(model_res)                  	

		## initialize label using ground truth
	for i in range(len(evl_data_set)):
		for j in range(len(evl_data_set[0])):
			res = evl_data_set[i][j][1]
			if res == ground_truth[0][title][i] or res == ground_truth[1][title][i] or res == ground_truth[2][title][i]:
				evl_data_set[i][j][3] = 1.0

	return evl_data_set






### using the raters to give the dicts initialization of labels
## 也可以直接fake所有的 label，因为目前我们人工的ground truth并不准确


### precision@k  k from 1 to n: represent the rank order
def precision(dicts, k):
	if dicts == None:
		print("Please load your test set!")
		return 0
	if k <= 0 or k > len(dicts):
		print("please input a valid value for k, which is from 1  to" + len(dicts))
		return 0
	
	sum = 0
	lens = len(dicts)
	## 对于sample中每一个rank为k的问答句pair precision@k = sum / sample中所有rank为k的问句总数
	for i in range(lens):
		#print(test_question_res[i][1][3])
		sum += dicts[i][k-1][3]

	precision = sum / len(dicts)
	return precision




## MAP value
def mAP(dicts, k):
	
	if dicts == None:
		print("Please load your test set!")
		return 0
	if k <= 0 or k > len(dicts):
		print("please input a valid value for k, which is from 1  to" + len(dicts))
		return 0
	
	sum1 = 0
	for i in range(len(dicts)):
		sum2 = 0
		Nq = 0
		for j in range(k):
			sum2 += precision(dicts, j+1) * dicts[i][j][3]
		for r in range(len(dicts[0])):
			Nq += dicts[i][r][3]

		if Nq == 0:
			sum1 += 0
		else:
			sum1 += sum2 / Nq

	return sum1 / len(dicts)


## MRR
def mRR(dicts):
	if dicts == None:
		print("Please load your test set!")
		return 0
	
	sum1 = 0
	for i in range(len(dicts)):
		for j in range(len(dicts[0])):
			if(dicts[i][j][3] == 1):
				sum1 += 1 / (j + 1)

	return sum1 / len(dicts)


## Z is a constant number


## nDCG
def nDCG(dicts, k):
	if dicts == None:
		print("Please load your test set!")
		return 0
	if k <= 0 or k > len(dicts):
		print("please input a valid value for k, which is from 1  to" + len(dicts))
		return 0
	
	sum1 = 0
	for i in range(len(dicts)):
		sum2 = 0
		for j in range(k):
			sum2 += (2**(dicts[i][j][3]) - 1) / np.log(1 + (i+1))
		
		sum1 += sum2 / (1.5 + 1 / np.log(3))
					 
		return sum1 / len(dicts)






if __name__ == '__main__':
	
	import sys
	args = sys.argv
	
	query = query_generate(args[3])
	ground = ground_truth(args[4], args[5], args[6])
	ground_truth = ground[0]
	title = ground[1]
	dicts = evaluation_data_gen(query, ground_truth, title)

	if args[1] == "precision":
		print("the precision@" + args[2] + " is ")
		print(precision(dicts, int(args[2])))
	elif args[1] == "MAP":
		print("the MAP of the sample for " + args[2] + "ranked is: " + str(mAP(dicts, int(args[2]))))
	elif args[1] == "MRR":
		print("the MRR of the sample" + args[2] + str(mRR(dicts)))
	elif args[1] == "nDCG":
		print("the nDCG of the sample for " + args[2] + " ranked is: " + str(nDCG(dicts, int(args[2]))))

