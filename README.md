
## CycleGAN+Unet discriminator
I2I transition 에서 in-the-wild 데이터셋 에 대한 baseline 성능을 평가한다.  
texture를 변환하는것이 아닌 구조를 바꾸는 task 자체가 어렵기 때문에 cycleGAN 학습은 제대로 수행되지 않는다.  
논문에서도  cat2dog 셋에 대하여 학습이 전혀 수행되지 않는다고 소개하고 있다.  
이를 해결하기 위해 U-net discriminator를 적용해보며 cycleGAN에 적용시 차이가 있는지 확인해 본다.  
  
* __현재까지의 실험결과 Unet 구조를 결합한 결과, 기존의 horse2zebra 셋에서도 제대로 학습이 수행되지 않는다.  
구조에 대한 코드리뷰가 필요할 것으로 판단된다.__

#### 현 실험 수행 사항
<p align="center">
<img style="margin:50px 0 10px 0" src="https://user-images.githubusercontent.com/40943064/128589211-6f2663ee-fab7-41b9-a852-2bd6a40644db.png"   width=600 />
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
