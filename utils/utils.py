# import


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/utils.org::*import][import:1]]
import csv
import re
import pickle
from functools import wraps

# import:1 ends here

# read_csv


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/utils.org::*read_csv][read_csv:1]]
def read_csv(file_path: str):
    """file_path: path of csv file.
return:
a list of dicts. each dict is a row of csv file."""
    with open(file_path, "r") as f:
        field_names = f.readline().replace("\n", "").split(",")
        ## dt = list(dict(i) for i in csv.DictReader(f, field_names))
        dt = list(csv.DictReader(f, field_names))
    return dt


# read_csv:1 ends here

# Register


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/utils.org::*Register][Register:1]]
class Register:
    """register anything with __name__ attribute into a special obj_name.

Example:

loss_funcs = Register()
@register
def mse(x, y):
    return np.mean(np.power(x - y, 2))

loss_funcs['mse']
"""

    def __init__(self):
        self.dict = {}

    def __call__(self, func):
        self.dict[func.__name__] = func
        # getattr(self, self.obj_name).update({func.__name__: func})
        @wraps(func)
        def wrapped_func(*args, **kargs):
            return func(*args, **kargs)

        return wrapped_func

    def __getitem__(self, key):
        return self.dict[key]


# Register:1 ends here

# RegSplit


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/utils.org::*RegSplit][RegSplit:1]]
class RegSplit:
    """Split a string by regular expression
Method: __init__, __call__
    """

    def __init__(self, pattern: str, exclude=[" "]):
        """pattern: string as regular expression"""
        import re

        self.pattern = pattern
        self.exclude = exclude
        self.spliter = re.compile(pattern)

    def __call__(self, string: str):
        finds = self.spliter.finditer(string)
        return [w[0] for w in finds if w[0] not in self.exclude]


# RegSplit:1 ends here

# load_by_name


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/utils.org::*load_by_name][load_by_name:1]]
def load_by_name(proc_path: str, obj_name: str):
    with open(f"{proc_path}/{obj_name}.pkl", "rb") as f:
        return pickle.load(f)


# load_by_name:1 ends here

# now


# [[file:~/Works/disease_name_normalization/cos_sim_cluster/utils/utils.org::*now][now:1]]
from datetime import datetime


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# now:1 ends here
