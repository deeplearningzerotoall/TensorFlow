# TensorFlow
Deep Learning Zero to All - TensorFlow

**Code Convention (10/7)**
Tensorflow 공식 홈페이지 가이드 따라하기: https://www.tensorflow.org/community/style_guide

1. Variable: **rnn_layer** / rnnLayer

2. tf.data: https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
  우선 make_initializable_iterator()
  진원님: CS20에 나오는 코드 샘플 (링크 공유, 전처리 tf.data 스타일은 우선은 이 스타일로)
  보섭님: 향후에 Estimator까지 포함하는 형태의 코드이면 좋겠다.

  데이터셋: train / validation
  추가로 모델을 불러서 test셋으로 예측하고 보는법 내용 추가

3. 전처리나 이쪽은 tf.keras쪽을 활용하자 (raw 스타일로 짜는거는 지양)

4. session에서 추가 기능?
  - config option 주고 열자 (gpu 모두 점유 방지)
  
5. 파일을 하나로 할 것인가? 몇개로 한 것인가?
  - ipynb는 1개로 제작하자
  - py는 제한은 없다.

6. tf.keras.layer + session으로 시작하자
  - 희망이나 추가로 샘플 estimator는 향후 추가 or not
  
7. 학습이 멈추는 조건을 어떻게 할 것인가?
 - Epoch로 조정
 - Ealrystopping
 - 특정 Condition (loss, acc, perplexity ....)
 의견1. 내용의 스타일에 따라 걸어도 되고 안해도 되고 (주로 복잡한 코드에 적용)

8. Tensorboard
 - 모든 코드에 굳이 넣을 필요는 없고, 기존 코드에 설명을 최신으로 추가\
 
9. name_scope vs. variable_scope
  - variable_scope은 2.0에서 사리진다고 함.
  - 추가 조사 필요 (keras에서 따로 control 하는게 있는가?)
  
10. layer 여러개 쌓을 때?
  - **cnn_layer1, cnn_layer2** vs. cnn_layer
