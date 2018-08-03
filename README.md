## Retrieval Model evaluation script

### Environment
```
Python 3.6+
numpy
pandas
json
jieba
jsonpickle
```
### Data
1. 语料库
```
put the corpus into the corpus folder
```
2. 测试问题集
```
put the test question data into the data folder, make sure the test data is txt file
```
3. ground truth 
```
put three gorund truth files in to the data folder(also txt file)
```

### Build inverted index
```
$ python simple_index_build.py corpus/
```
You will get two generated file: inverted_index.josn and forward_index.json

### calculate the evaluation attribute
1. precision@k (k can be [1,3])
```
$ python test.py precision 1 data/test_question.txt data/ground_truth1.txt data/ground_truth2.txt data/ground_truth3.txt
```
2. MAP@k
```
python test.py MAP 3 data/test_question.txt data/ground_truth1.txt data/ground_truth2.txt data/ground_truth3.txt
```
3. MRR
```
python test.py MRR == data/test_question.txt data/ground_truth1.txt data/ground_truth2.txt data/ground_truth3.txt
```
4. nDCG@k
```
python test.py nDCG 3 data/test_question.txt data/ground_truth1.txt data/ground_truth2.txt data/ground_truth3.txt
```
 
 
 
 
