# 00.helloworld.py
#
# This example will create a window where you 
# should only see a gaussian noise pattern

import nvisii

nvisii.initialize()

while (not nvisii.should_window_close()): pass

nvisii.deinitialize()