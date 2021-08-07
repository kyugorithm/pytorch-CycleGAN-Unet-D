
## CycleGAN에 Unet discriminator 구조를 결합하여 I2I trainsition 성능을 파악한다.

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
