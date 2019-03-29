# 모두를 위한 딥러닝 시즌 2 : 모두가 만드는 모두를 위한 딥러닝

모두가 만드는 모두를 위한 딥러닝 시즌 2에 오신 여러분들 환영합니다. 

## Getting Started

아래 링크에서 슬라이드와 영상을 통해 학습을 시작할 수 있습니다.

* Slide: http://bit.ly/2LQMKvk
* YouTube: http://bit.ly/2HHrybT

### Docker 사용자를 위한 안내

동일한 실습 환경을 위해 docker를 사용하실 분은  [docker_user_guide.md](docker_user_guide.md) 파일을 참고하세요! :)

### Install Requirements

```bash
pip install -r requirements.txt
```

---

## TensorFlow

Deep Learning Zero to All - TensorFlow

모든 코드는 Tensorflow 1.12(stable)를 기반으로 작성했으며 Tensorflow 2.0이 출시되는 대로 추후 반영할 예정입니다.

## Standarad of Code

코드는 Tensorflow 공식 홈페이지 권장에 따라 Keras + Eager로 작성했으며 

Session 버전은 code_session_version / Keras 버전은 other에서 확인하실 수 있습니다.

## Contributions/Comments

언제나 여러분들의 참여를 환영합니다. Comments나 Pull requests를 남겨주세요.

We always welcome your comments and pull requests.

## 목차

### PART 1: Basic Machine Learning

* Lec 01: 기본적인 Machine Learning의 용어와 개념 설명
* Lab 01: (추가 예정)
* Lec 02: Simple Linear Regression
* Lab 02: Simple Linear Regression를 TensorFlow로 구현하기
* Lec 03: Linear Regression and How to minimize cost
* Lab 03: Linear Regression and How to minimize cost를 TensorFlow로 구현하기
* Lec 04: Multi-variable Linear Regression
* Lab 04: Multi-variable Linear Regression를 TensorFlow로 구현하기
* Lec 05-1: Logistic Regression/Classification의 소개
* Lec 05-2: Logistic Regression/Classification의 cost 함수, 최소화
* Lab 05-3: Logistic Regression/Classification를 TensorFlow로 구현하기
* Lec 06-1: Softmax Regression: 기본 개념 소개
* Lec 06-2: Softmax Classifier의 cost 함수
* Lab 06-1: Softmax classifier를 TensorFlow로 구현하기
* Lab 06-2: Fancy Softmax classifier를 TensorFlow로 구현하기
* Lab 07-1: Application & Tips: 학습률(Learning Rate)과 데이터 전처리(Data Preprocessing)
* Lab 07-2-1: Application & Tips: 오버피팅(Overfitting) & Solutions
* Lab 07-2-2: Application & Tips: 학습률, 전처리, 오버피팅을 TensorFlow로 실습
* Lab 07-3-1: Application & Tips: Data & Learning
* Lab 07-3-2: Application & Tips: 다양한 Dataset으로 실습

### PART 2: Basic Deep Learning

* Lec 08-1: 딥러닝의 기본 개념: 시작과 XOR 문제
* Lec 08-2: 딥러닝의 기본 개념 2: Back-propagation 과 2006/2007 '딥'의 출현
* Lec 09-1: XOR 문제 딥러닝으로 풀기
* Lec 09-2: 딥넷트웍 학습 시키기 (backpropagation)
* Lab 09-1: Neural Net for XOR
* Lab 09-2: Tensorboard (Neural Net for XOR)
* Lab 10-1: Sigmoid 보다 ReLU가 더 좋아
* Lab 10-2: Weight 초기화 잘해보자
* Lab 10-3: Dropout
* Lab 10-4: Batch Normalization

### PART 3: Convolutional Neural Network

* Lec 11-1: ConvNet의 Conv 레이어 만들기
* Lec 11-2: ConvNet Max pooling 과 Full Network
* Lec 11-3: ConvNet의 활용 예
* Lab 11-0-1: CNN Basic: Convolution
* Lab 11-0-2: CNN Basic: Pooling
* Lab 11-1: mnist cnn keras sequential eager
* Lab 11-2: mnist cnn keras functional eager
* Lab-11-3: mnist cnn keras subclassing eager
* Lab-11-4: mnist cnn keras ensemble eager
* Lab-11-5: mnist cnn best keras eager

### PART 4: Recurrent Neural Network

* Lec 12: NN의 꽃 RNN 이야기
* Lab 12-0: rnn basics
* Lab 12-1: many to one (word sentiment classification)
* Lab 12-2: many to one stacked (sentence classification, stacked)
* Lab 12-3: many to many (simple pos-tagger training)
* Lab 12-4: many to many bidirectional (simpled pos-tagger training, bidirectional)
* Lab 12-5: seq to seq (simple neural machine translation)
* Lab 12-6: seq to seq with attention (simple neural machine translation, attention)
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




