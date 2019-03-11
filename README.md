# 모두를 위한 딥러닝 시즌2 : 모두가 만드는 모두를 위한 딥러닝

Sung Kim 교수님의 모두를 위한 딥러닝이 돌아왔습니다!

이 강의는 2016년 Sung Kim 교수님이 만드신 '모두를 위한 딥러닝(https://hunkim.github.io/ml/)' 의 개정판이자 후속작입니다.

"알파고와 이세돌의 경기를 보면서 이제 머신 러닝이 인간이 잘 한다고 여겨진 직관과 의사 결정능력에서도 충분한 데이타가 있으면 어느정도 또는 우리보다 더 잘할수도 있다는 생각을 많이 하게 되었습니다. Andrew Ng 교수님이 말씀하신것 처럼 이런 시대에 머신 러닝을 잘 이해하고 잘 다룰수 있다면 그야말로 'Super Power'를 가지게 되는 것이 아닌가 생각합니다.

더 많은 분들이 머신 러닝과 딥러닝에 대해 더 이해하고 본인들의 문제를 이 멋진 도구를 이용해서 풀수 있게 하기위해 비디오 강의를 준비하였습니다. 더 나아가 이론에만 그치지 않고 머신러닝을 위한 오픈소스인 구글이 공개한 TensorFlow와 페이스북이 공개한 Pytorch를 이용해서 이론을 구현해 볼수 있도록 하였습니다.

수학이나 컴퓨터 공학적인 지식이 없이도 쉽게 볼수 있도록 만들려고 노력하였습니다."

-홍콩과기대 컴퓨터공학 교수 김성훈(Sung Kim)

## TensorFlow
Deep Learning Zero to All - TensorFlow

여기는 TensorFlow 버전 Github 문서입니다.

현재는 Tensorflow 1.12(stable)를 기반으로 작성했으며 Tensorflow 2.0이 출시되는 대로 추후 반영할 예정입니다.


## Standarad of Code

코드는 Tensorflow 공식 홈페이지 권장에 따라 Keras + Eager로 작성했으며 

Session 버전은 code_session_version / Keras 버전은 other에서 확인하실 수 있습니다.


## Install Requirements

```bash
pip install -r requirements.txt
```

## Contributions/Comments

언제나 여러분들의 참여를 환영합니다. Comments나 Pull requests를 남겨주세요

We always welcome your comments and pull requests.

------------------------------------
### Docker 사용자를 위한 안내

[docker_user_guide.md](docker_user_guide.md) 파일을 참고하세요! :)

### 목차
* Lec 01: 기본적인 Machine Learning 의 용어와 개념 설명
* Lab 01: (추가예정)
* Lec 02: Simple Linear Regression
* Lab 02: Simple Linear Regression 를 TensorFlow 로 구현하기
* Lec 03: Linear Regression and How to minimize cost
* Lab 03: Linear Regression and How to minimize cost 를 TensorFlow 로 구현하기
* Lec 04: Multi-variable Linear Regression
* Lab 04: Multi-variable Linear Regression 를 TensorFlow 로 구현하기
* Lec 05-1: Logistic Regression/Classification 의 소개
* Lec 05-2: Logistic Regression/Classification 의 cost 함수, 최소화
* Lab 05-3: Logistic Regression/Classification 를 TensorFlow 로 구현하기
* Lec 06-1: Softmax Regression: 기본 개념소개
* Lec 06-2: Softmax Classifier의 cost함수
* Lab 06-1: Softmax classifier 를 TensorFlow 로 구현하기
* Lab 06-2: Fancy Softmax classifier 를 TensorFlow 로 구현하기
* Lec 07-1: Application & Tips: 학습률(Learning Rate)과 데이터 전처리(Data Preprocessing)
* Lec 07-2: Application & Tips: 오버피팅(Overfitting) & Solutions
* Lab 07-1: Application & Tips: 학습률, 전처리, 오버피팅을 TensorFlow 로 실습
* Lec 07-3: Application & Tips: Data & Learning
* Lab 07-2: Application & Tips: 다양한 Dataset 으로 실습
* Lec 08-1: 딥러닝의 기본 개념: 시작과 XOR 문제
* Lec 08-2: 딥러닝의 기본 개념2: Back-propagation 과 2006/2007 '딥'의 출현
* Lec 09-1: XOR 문제 딥러닝으로 풀기
* Lec 09-2: 딥넷트웍 학습 시키기 (backpropagation)
* Lab 09-1: Neural Net for XOR
* Lab 09-2: Tensorboard (Neural Net for XOR)
* Lab 10-1: Sigmoid 보다 ReLU가 더 좋아
* Lab 10-2: Weight 초기화 잘해보자
* Lab 10-3: Dropout
* Lab 10-4: Batch Normalization
* Lec 11-1: ConvNet의 Conv 레이어 만들기
* Lec 11-2: ConvNet Max pooling 과 Full Network
* Lec 11-3: ConvNet의 활용예
* Lab 11-0: CNN Basic: Convolution
* Lab 11-0: CNN Basic: Pooling
* Lab 11-1: mnist cnn keras sequential eager
* Lab 11-2: mnist cnn keras functional eager
* Lab-11-3: mnist cnn keras subclassing eager
* Lab-11-4: mnist cnn keras ensemble eager
* Lab-11-5: mnist cnn best keras eager
* Lec 12: NN의 꽃 RNN 이야기
* [Lab 12-0: rnn basics](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-12-0-rnn-basics-keras-eager.ipynb)
* [Lab 12-1: many to one (word sentiment classification)](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-12-1-many-to-one-keras-eager.ipynb)
* [Lab 12-2: many to one stacked (sentence classification, stacked)](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-12-2-many-to-one-stacking-keras-eager.ipynb)
* [Lab 12-3: many to many (simple pos-tagger training)](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-12-3-many-to-many-keras-eager.ipynb)
* [Lab 12-4: many to many bidirectional (simpled pos-tagger training, bidirectional)](https://nbviewer.jupyter.org/github/deeplearningzerotoall/TensorFlow/blob/master/lab-12-4-many-to-many-bidirectional-keras-eager.ipynb)
* [Lab 12-5: seq to seq (simple neural machine translation)](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-12-5-seq-to-seq-keras-eager.ipynb)
* [Lab 12-6: seq to seq with attention (simple neural machine translation, attention)](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/lab-12-6-seq-to-seq-with-attention-keras-eager.ipynb)

--------------------------

### 함께 만든 이들

Main Instructor
* Prof. Kim (https://github.com/hunkim)

Main Creator
* 김보섭 (https://github.com/aisolab)
* 김수상 (https://github.com/healess)
* 김준호 (https://github.com/taki0112)
* 신성진 (https://github.com/aiscientist)
* 이승준 (https://github.com/FinanceData)
* 이진원 (https://github.com/jwlee-ml)

Docker Developer
* 오상준 (https://github.com/juneoh)

Support
* 네이버 커넥트재단 : 이효은, 장지수, 임우담




