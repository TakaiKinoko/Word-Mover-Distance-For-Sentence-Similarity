# Semantic Similarity Library: model interface
#
# Copyright (C) 2019-2020 MotleyWorks
# Author: Fang Han <fang@buymecoffee.co>

"""
Define api for all models under /models.
TODO use formal interface when code base grows big

The ModelInterface defines the contract with which all Model classes in the models package should comply.

A concrete Model class should have the following methods:




Be sure to call:
    issubclass(<Model>, ModelInterface)
to test if the Model has successfully implemented all the abstract methods above.

Ref: https://realpython.com/python-interface/#informal-interfaces
"""


class ModelMeta(type):
    """A Model metaclass that will be used for model class creation.
    """
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'compute_pair_sim') and
                callable(subclass.compute_pair_sim) and
                hasattr(subclass, 'compute_sim_list') and
                callable(subclass.compute_sim_list) and
                hasattr(subclass, 'compute_sim_list_batch') and
                callable(subclass.compute_sim_list_batch) and
                hasattr(subclass, 'evaluate_model') and
                callable(subclass.evaluate_model))


class ModelInterface(metaclass=ModelMeta):
    """
    This interface is used for concrete classes to inherit from.
    There is no need to define the ModelMeta methods as any class
    as they are implicitly made available via .__subclasscheck__().
    """
    pass
