import itertools


# [1,2,3,4] -> [[1,2],[3,4]]
def group(l, group_size):
    args = [iter(l)] * group_size
    return itertools.zip_longest(*args)

