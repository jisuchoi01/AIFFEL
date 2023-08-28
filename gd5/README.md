# AIFFEL Campus Online 5th Code Peer Review Templete
- 코더 : 최지수
- 리뷰어 : 조준규


# PRT(PeerReviewTemplate) 
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  > U-Net++ 모델에 대한 구현이 완성되지 않았다.
- [ ] 주석을 보고 작성자의 코드가 이해되었나요?
  > KITTI dataset 외에는 주석이 많이 작성되어 있진 않았다.
- [X] 코드가 에러를 유발할 가능성이 없나요?
  > 전체적으로 에러가 일어날 부분은 크게 없었다.
- [X] 코드 작성자가 코드를 제대로 이해하고 작성했나요?
  > U-Net을 알맞게 잘 구현한 것으로 보아 제대로 이해했다고 생각한다.
- [ ] 코드가 간결한가요?
  > 모델을 구현할 때 conv block을 따로 구현하여 사용하지는 않아서 간결하진 않았다.

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 코드 이해도 예시
# MaxPooling과 UpSampling을 적절히 잘 사용하여 U-Net을 알맞게 구현함.
def build_model(input_shape=(224, 224, 3)):
    model = None
    inputs = Input(input_shape)
    c1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    c1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(c1)
    p1 =layers.MaxPooling2D(pool_size=(2, 2))(c1)

    ...

    up4 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(dr5))
    merge4 = concatenate([c1,up4], axis = 3)
    c9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(merge4)
    c9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same')(c9)

    c10 = layers.Conv2D(1, 1, activation = 'sigmoid')(c9)
    model = Model(inputs = inputs, outputs=[c10])
    
    return model
```

# 참고링크 및 코드 개선
```python
# U-Net++가 겉보기에는 복잡해보여도 skip connection만 잘 생각하면 쉽게 구현할 수 있습니다.
# 규칙을 잘 찾아서 U-Net++ 구현에 꼭 성공하시길 바라겠습니다.😁
```
