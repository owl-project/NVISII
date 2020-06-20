import noise
import numpy as np
import randomcolor
# from scipy.misc import toimage

rc = randomcolor.RandomColor()
c1 = eval(str(rc.generate(format_='rgb',luminosity='bright')[0])[3:])
c2 = eval(str(rc.generate(format_='rgb',luminosity='bright')[0])[3:])
c3 = eval(str(rc.generate(format_='rgb',luminosity='bright')[0])[3:])


shape = (1024,1024)
scale = 100.0
octaves = 6
persistence = 0.5
lacunarity = 2.0

world2 = (1024,1024,3) 

world = np.zeros(shape)
world2 = np.zeros(world2)

for i in range(shape[0]):
    for j in range(shape[1]):
        world[i][j] = noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=1024, 
                                    repeaty=1024, 
                                    base=0)
        # if world[i][j] < 0: 
        # 	world2[i][j] = c1
        # elif world[i][j] > 0 and world[i][j] < 0.2 : 
        # 	world2[i][j] = c2
        # else:
        # 	world2[i][j] = c3



world = (world + abs(world.min())) / (world.max()+abs(world.min()))
world *= 255




world = world.reshape(shape[0],shape[1],1)
img = np.concatenate([world,world,world],axis=2)

# print(img.shape)
# print(img.min())
# print(img.max())

from PIL import Image
img = Image.fromarray((img).astype(np.uint8))
img.save("tex.png")


# TRYING 3D noise




