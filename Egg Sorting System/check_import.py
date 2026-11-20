import sys, os
sys.path.insert(0, r'C:\laragon\www\Egg-SortingSystem\Egg Sorting System')
import modules.shape_size_detector as ssd
print("ssd.__file__:", ssd.__file__)
import inspect
print("measure_egg_shape_and_size source lines:")
lines, _ = inspect.getsourcelines(ssd.measure_egg_shape_and_size)
print("".join(lines[:20]))
