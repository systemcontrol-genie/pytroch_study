import math
import torch
import matplotlib.pyplot as plt

x =torch.linspace(-math.pi , math.pi , 1000) # -pi 부터 pi 까지 1000개의 데이터 가져온다.

y = torch.sin(x) # sin 그래프 출력

a = torch.randn(()) # random 값을 a에 정의 한다.
b = torch.randn(()) # random 값을 b에 정의 한다.
c = torch.randn(()) # random 값을 c에 정의 한다.
d = torch.randn(()) # random 값을 b에 정의 한다.

y_random = a*x**3+ b*x**2+ c*x +d # random 함수를 통하여 그래프 출력

plt.subplot(2,1,1) # 첫번째 그래프에 y_true 그래프 출력
plt.title("y_true")
plt.plot(x,y)

plt.subplot(2,1,2) # 두번쨰 그래프에 y_pred 그래프 출력
plt.title("y_pred")
plt.plot(x, y_random)

plt.show()

