
## CycleGAN+Unet discriminator
I2I transition 에서 in-the-wild 데이터셋 에 대한 baseline 성능을 평가한다.  
texture를 변환하는것이 아닌 구조를 바꾸는 task 자체가 어렵기 때문에 cycleGAN 학습은 제대로 수행되지 않는다.  
논문에서도  cat2dog 셋에 대하여 학습이 전혀 수행되지 않는다고 소개하고 있다.  
 U-net discriminator를 적용해보며 cycleGAN에 적용시 유의미한 차이가 있는지 확인해 본다.  

**update : 2021-08-08**
Blob이 생성되는 문제를 해결하였다.  
1) Relu로 구현되어있는 loss를 일반 ls loss로 변경하였고  
2) D와 G의 상대적 Loss가 너무 차이나지 않도록 값을 조절하였다.  
그러나 생성물을 보면 영역 분리가 명확하지 않아 추가학습 확인이 필요하며 / 추가 Capacity에 대한 결과확인이 필요하다.  
cycleGAN의 기본 세팅은 BatchNorm으로 구성되어있으나 minibatch가 1인 경우 instance norm으로 수행된다.  
I2I translation task에서는 instance norm이 효과적이기때문에 batch size를 올렸을때 유의해야한다.
추가로, 
* 향후 cat2dog task에 대해서 해당 모델이 어떤 결과를 보일지 확인이 필요하다.
**update : 2021-08-08**
Receptive field에 대한 고민을 해 보았다. Discriminator를 설계할 때 cycleGAN은 patchGAN을 이용한다.  
이는 receptive field를 넓게 가져가기 위함이다. 즉, 픽셀이 아닌 꽤 큰 영역을 둘러 보는 것이  
이미지 패턴을 이해하고 이를 구분하는 것이 더 의미가 있다는 것이다. pixelGAN (1x1 patchGAN) 에서 수용 영역이 점점 넓어지는  
case (3layer ~ 5layer)를 실험해본 결과 모두 동일하게 잘 나온다. 그러나 dog2cat dataset에는 전혀 효과가 없었다.

#### 현 실험 수행 사항
<p align="center">
<img style="margin:50px 0 10px 0" src="https://user-images.githubusercontent.com/40943064/128622709-36dff1bf-01a6-4f47-bec7-f9a9b7ed24d8.png"   width=600 />
 

</p>
#### 수정 영역
### Backward of discriminator
1) AS-IS  
<p align="center">
<img style="margin:50px 0 10px 0" src="https://user-images.githubusercontent.com/40943064/128587308-f0a60e04-929c-4c69-9953-ae15ad8f7e6a.png"   width=300 />
</p>
  
  
2) TO-BE  

<p align="center">
<img style="margin:50px 0 10px 0" src="https://user-images.githubusercontent.com/40943064/128587299-20565248-899c-444d-bea7-01f582a43b09.png"   width=600 />
</p>

### Backward of generator
<p align="center">
<img style="margin:50px 0 10px 0" src="https://user-images.githubusercontent.com/40943064/128587446-6da890af-9c9f-4106-a16b-45a13b8fb60e.png"   width=600 />
</p>
