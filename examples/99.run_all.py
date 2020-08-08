import os 
import glob
import shutil
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--outf',
                    default='all_examples',
                    help = 'folder to output the examples')
opt = parser.parse_args()

if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.mkdir(opt.outf)
    print(f'created folder {opt.outf}/')


for test in sorted(glob.glob("*.py")):
    if  test == '99.run_all.py' or\
        test == '00.helloworld.py' or\
        '15' in test:
        continue
    # 
    print(f'running {test}')
    subprocess.call(
        [
            "python",test
        ]
    )
    if "pybullet" in test:
        shutil.move('output.mp4',f'{opt.outf}/{test.replace(".py",".mp4")}')
    else:
        if os.path.exists('tmp.png'):
            shutil.move('tmp.png',f'{opt.outf}/{test.replace(".py",".png")}')
        if os.path.exists('tmp.hdr'):
            shutil.move('tmp.hdr',f'{opt.outf}/{test.replace(".py",".hdr")}')
        if os.path.exists('metadata/'):
            shutil.move('metadata/',f'{opt.outf}/{test.replace(".py","/")}')

    # break