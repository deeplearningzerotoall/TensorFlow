# TensorFlow
Deep Learning Zero to All - TensorFlow

**Code Convention (10/7)**

1. Variable: **rnn_layer** / rnnLayer

2. tf.data: https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
  우선 make_initializable_iterator()
  진원님: CS20에 나오는 코드 샘플 (링크 공유, 전처리 tf.data 스타일은 우선은 이 스타일로)
  보섭님: 향후에 Estimator까지 포함하는 형태의 코드이면 좋겠다.

  데이터셋: train / validation / test

3. 전처리나 이쪽은 tf.keras쪽을 활용하자 (raw 스타일로 짜는거는 지양)

4. session에서 추가 기능?
  - config option 주고 열자 (gpu 모두 점유 방지)
  - earlystopping?
  - tensorboard?
  
5. 
