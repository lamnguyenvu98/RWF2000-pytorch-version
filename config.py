import yaml
from dotmap import DotMap

def read_args(filename):
    with open(filename) as f:
        cf = yaml.safe_load(f)
    f.close()

    args = DotMap(cf)
    return args
