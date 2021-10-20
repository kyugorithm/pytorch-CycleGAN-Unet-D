**아래 3가지 결과는 학습량이 달라 동등한 기준으로 볼 수는 없으나 4 down layer G의 경우 25epoch부터 변화가 확연히 발생했음**
## 4 Layer로 증가시켰을 경우 변화가 상대적으로 큼(40epochs ==500,000 iterations)
- 최초 설계한 CAM1 loss가 상대적으로 크게 감소
- 형태 강아지와 같은 형상의 이미지가 많이 발생하나 개별 품질은 좋지 못함 (품질성능 향상 방법?:G encoder를 다양한 해상도로?)
![image](https://user-images.githubusercontent.com/40943064/138091171-a35142a5-4517-43d5-a516-6bde67e0e1b8.png)  

![image](https://user-images.githubusercontent.com/40943064/138091127-e7b35588-875e-4ebc-a98c-c3d9550729db.png)  
![image](https://user-images.githubusercontent.com/40943064/138091141-10488cf4-72b9-403e-b0f0-6d568b4d906e.png)  

## 참고1. G down 2-layer(25epochs ==312,500 iterations)
![image](https://user-images.githubusercontent.com/40943064/138091720-c1975941-5c16-4d0d-b931-4da4335780d0.png)

## 참고2. CycleGAN Baseline(PatchGAN Discriminator 5layers ; 9epoch==112,000 iterations)
![image](https://user-images.githubusercontent.com/40943064/138092270-446252ac-5929-4501-9b3a-f198617fe855.png)
