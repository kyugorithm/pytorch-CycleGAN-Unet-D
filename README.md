
## CycleGAN+Unet discriminator
I2I transition 에서 in-the-wild 데이터셋 에 대한 baseline 성능을 평가한다.  
texture를 변환하는것이 아닌 구조를 바꾸는 task 자체가 어렵기 때문에 cycleGAN 학습은 제대로 수행되지 않는다.  
논문에서도  cat2dog 셋에 대하여 학습이 전혀 수행되지 않는다고 소개하고 있다.  
 U-net discriminator를 적용해보며 cycleGAN에 적용시 차이가 있는지 확인해 본다.  
* 검정색 점이 생성되는 문제를 해결하였다. 
* 1) Relu로 구현되어있는 loss를 일반 ls loss로 변경하였고
* 2) D와 G의 상대적 Loss가 너무 차이나지 않도록 값을 조절하였다. 
* 다만, 생성물을 보면 영역 분리가 명확하지 않아 추가 학습시 어떻게 진행될지 확인이 필요하며 capacity를 늘려 학습을 수행해보아야 한다.
* 향후 cat2dog task에 대해서 해당 모델이 어떤 결과를 보일지 확인이 필요하다.

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
