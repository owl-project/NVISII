import visii 
import numpy as np 
from PIL import Image 
import time 


visii.initialize_headless()

# time to initialize this is a bug
time.sleep(3)


x = np.array(visii.read_frame_buffer()).reshape(512,512,4)
img = Image.fromarray((x*255).astype(np.uint8))

# You should see a noise image, like gaussian noise. 
img.save("tmp.png")

visii.cleanup()

