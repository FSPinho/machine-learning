from copy import deepcopy


class Cloneable:
    def clone(self, **kwargs):
        clone = deepcopy(self)

        for key, val in kwargs.items():
            setattr(clone, key, val)

        return clone
