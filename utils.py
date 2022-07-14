from inspect import isfunction

def exists(x):
    return x is not None

def default(val,d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num,divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr