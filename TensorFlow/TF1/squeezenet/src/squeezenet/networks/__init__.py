from squeezenet.networks.squeezenet import Squeezenet
from squeezenet.networks.squeezenet import Squeezenet_CIFAR
from squeezenet.networks.squeezenet import Squeezenet_Tiny

catalogue = dict()


def register(cls):
    catalogue.update({cls.name: cls})


register(Squeezenet)
register(Squeezenet_CIFAR)
register(Squeezenet_Tiny)
