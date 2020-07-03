import numpy as np
import matplotlib.pyplot as plt


def diff_eq(dd, d, d0):
# ddx = agent_selects
    d = d + dd
    d0 = d0 + d
    return dd, d, d0

ddx, dx, x = 0, 0, 0
ddz, dz, z = 0, 0, 0

for i in range(110):
    if i < 60:
        ddz, dz, z = diff_eq(0.1, dz, z)
        #ddx, dx, x = diff_eq(-0.01, dx, x)
    elif i >= 60 and i < 90:
        ddz, dz, z = diff_eq(0.05, dz, z)
        ddx, dx, x = diff_eq(0.003, dx, x)
    else:
        ddz, dz, z = diff_eq(-0.3, dz, z)
        ddx, dx, x = diff_eq(0.005, dx, x)
    #z = z + 0.3
    
    plt.plot(x, z, 'ob')

plt.gca().invert_yaxis()
plt.xlabel("Cross Section")
plt.ylabel("TVD")
plt.show()


    