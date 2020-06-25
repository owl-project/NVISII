# ViSii examples

This folder contains different examples showing different capabilities for visii. 
The first example uses interactive mode, whereas the rest of the examples 
use headless mode and save local images (look for `tmp.png` in the `examples/` 
folder). 

Please not that if you have more than one GPU, you might run into problems if they 
are not the same. Please use only one GPU with the following `CUDA_VISIBLE_DEVICES=0`. 
For example you could run: 
```
CUDA_VISIBLE_DEVICES=0 python 1.simple_scene.py
```

## 3.pybullet.py
This example creates a video in a folder, you are going to need `ffmpeg` to generate 
the final video, which is called `output.mp4` and it is located in the `examples/` folder.


## 4.load_obj_file.py
This script assumes you have downloaded the content for the demos using `sh download_content.sh`. 
You can also use your own obj file, see the arguments to the script. 


## Notes
All these examples were developed and tested on Ubuntu 18.04 with cuda 10.2, NVIDIA drivers
440.100 and using a NVIDIA RTX 2080 ti. 


## TODO
- loading textures
- loading an obj scene 
- exporting data
- procedural texture

