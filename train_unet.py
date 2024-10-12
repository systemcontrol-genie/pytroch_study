import math
import matplotlib.pyplot as plt
import torch

x = torch.linspace(-math.pi , math.pi , 1000)

y = torch.sin(x)

a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

y_random = a*x**3 + b*x**2 + c*x + d

plt.subplot(2,1,1)
plt.title("y_true")
plt.plot(x, y)

plt.subplot(2,1,2)
plt.title('y_pred')
plt.plot(x,y_random)

plt.show()