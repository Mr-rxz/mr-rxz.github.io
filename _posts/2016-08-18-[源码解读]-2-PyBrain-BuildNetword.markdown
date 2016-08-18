---
layout: post
title:  "[源码解读] 2 PyBrain BuildNetword"
date:   2016-08-18 23:34:41
category:  "源码解读"
---

In PyBrain, networks are composed of Modules which are connected with Connections. You can think of a network as a directed acyclic graph, where the nodes are Modules and the edges are Connections. This makes PyBrain very flexible but it is also not necessary in all cases.

# The buildNetwork Shortcut

Thus, there is a simple way to create networks, which is the buildNetwork shortcut:


{% highlight python %}

from pybrain.tools.shortcuts import buildNetwork
net = buildNetwork(2, 3, 1)

{% endhighlight %}

This call returns a network that has two inputs, three hidden and a single output neuron. In PyBrain, these layers are Module objects and they are already connected with FullConnection objects.

# The Source Code


{% highlight python %}

__author__ = 'Tom Schaul and Thomas Rueckstiess'


from itertools import chain
import logging 
from sys import exit as errorexit
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer
from pybrain.structure.connections import FullConnection, IdentityConnection

try:
    from arac.pybrainbridge import _RecurrentNetwork, _FeedForwardNetwork
except ImportError, e:
    logging.info("No fast networks available: %s" % e)


class NetworkError(Exception): pass


def buildNetwork(*layers, **options):
    """Build arbitrarily deep networks.
    
    `layers` should be a list or tuple of integers, that indicate how many 
    neurons the layers should have. `bias` and `outputbias` are flags to 
    indicate whether the network should have the corresponding biases; both
    default to True.
        
    To adjust the classes for the layers use the `hiddenclass` and  `outclass`
    parameters, which expect a subclass of :class:`NeuronLayer`.
    
    If the `recurrent` flag is set, a :class:`RecurrentNetwork` will be created, 
    otherwise a :class:`FeedForwardNetwork`.
    
    If the `fast` flag is set, faster arac networks will be used instead of the 
    pybrain implementations."""
    # options
    opt = {'bias': True,
           'hiddenclass': SigmoidLayer,
           'outclass': LinearLayer,
           'outputbias': True,
           'peepholes': False,
           'recurrent': False,
           'fast': False,
    }
    for key in options:
        if key not in opt.keys():
            raise NetworkError('buildNetwork unknown option: %s' % key)
        opt[key] = options[key]
    
    if len(layers) < 2:
        raise NetworkError('buildNetwork needs 2 arguments for input and output layers at least.')
        
    # Bind the right class to the Network name
    network_map = {
        (False, False): FeedForwardNetwork,
        (True, False): RecurrentNetwork,
    }
    try:
        network_map[(False, True)] = _FeedForwardNetwork
        network_map[(True, True)] = _RecurrentNetwork
    except NameError:
        if opt['fast']:
            raise NetworkError("No fast networks available.")
    if opt['hiddenclass'].sequential or opt['outclass'].sequential:
        if not opt['recurrent']:
            # CHECKME: a warning here?
            opt['recurrent'] = True
    Network = network_map[opt['recurrent'], opt['fast']]
    n = Network()
    # linear input layer
    n.addInputModule(LinearLayer(layers[0], name='in'))
    # output layer of type 'outclass'
    n.addOutputModule(opt['outclass'](layers[-1], name='out'))
    if opt['bias']:
        # add bias module and connection to out module, if desired
        n.addModule(BiasUnit(name='bias'))
        if opt['outputbias']:
            n.addConnection(FullConnection(n['bias'], n['out']))
    # arbitrary number of hidden layers of type 'hiddenclass'
    for i, num in enumerate(layers[1:-1]):
        layername = 'hidden%i' % i
        n.addModule(opt['hiddenclass'](num, name=layername))
        if opt['bias']:
            # also connect all the layers with the bias
            n.addConnection(FullConnection(n['bias'], n[layername]))
    # connections between hidden layers
    for i in range(len(layers) - 3):
        n.addConnection(FullConnection(n['hidden%i' % i], n['hidden%i' % (i + 1)]))
    # other connections
    if len(layers) == 2:
        # flat network, connection from in to out
        n.addConnection(FullConnection(n['in'], n['out']))
    else:
        # network with hidden layer(s), connections from in to first hidden and last hidden to out
        n.addConnection(FullConnection(n['in'], n['hidden0']))
        n.addConnection(FullConnection(n['hidden%i' % (len(layers) - 3)], n['out']))
    
    # recurrent connections
    if issubclass(opt['hiddenclass'], LSTMLayer):
        if len(layers) > 3:
            errorexit("LSTM networks with > 1 hidden layers are not supported!")
        n.addRecurrentConnection(FullConnection(n['hidden0'], n['hidden0']))

    n.sortModules()
    return n
    

def _buildNetwork(*layers, **options):
    """This is a helper function to create different kinds of networks.

    `layers` is a list of tuples. Each tuple can contain an arbitrary number of
    layers, each being connected to the next one with IdentityConnections. Due 
    to this, all layers have to have the same dimension. We call these tuples
    'parts.'
    
    Afterwards, the last layer of one tuple is connected to the first layer of 
    the following tuple by a FullConnection.
    
    If the keyword argument bias is given, BiasUnits are added additionally with
    every FullConnection. 

    Example:
    
        _buildNetwork(
            (LinearLayer(3),),
            (SigmoidLayer(4), GaussianLayer(4)),
            (SigmoidLayer(3),),
        )
    """
    bias = options['bias'] if 'bias' in options else False
    
    net = FeedForwardNetwork()
    layerParts = iter(layers)
    firstPart = iter(layerParts.next())
    firstLayer = firstPart.next()
    net.addInputModule(firstLayer)
    
    prevLayer = firstLayer
    
    for part in chain(firstPart, layerParts):
        new_part = True
        for layer in part:
            net.addModule(layer)
            # Pick class depending on whether we entered a new part
            if new_part:
                ConnectionClass = FullConnection
                if bias:
                    biasUnit = BiasUnit('BiasUnit for %s' % layer.name)
                    net.addModule(biasUnit)
                    net.addConnection(FullConnection(biasUnit, layer))
            else:
                ConnectionClass = IdentityConnection
            new_part = False
            conn = ConnectionClass(prevLayer, layer)
            net.addConnection(conn)
            prevLayer = layer
    net.addOutputModule(layer)
    net.sortModules()
    return net

{% endhighlight %}

# 名次解释
RecurrentNetwork：递归网络
FeedForwardNetwork：前馈神经网络

# 关键点解释

>1- def buildNetwork (*layers, **options)中“*”与“**”的意义。

{% highlight python %}
def func(*args):print(args)
{% endhighlight %}
当用func(1,2,3) 调用函数时,参数args就是元组(1,2,3)

{% highlight python %}
def func(**args):print(args)
{% endhighlight %}
当用func(a=1,b=2) 调用函数时,参数args将会是字典{'a':1,'b':2}

{% highlight python %}
def func(*args1, **args2):
    print(args1)
    print(args2)
{% endhighlight %}
当调用func(1,2, a=1,b=2)时，打印：(1,2) {'a': 1, 'b': 2}
当调用func(a=1,b=2)时，打印：() {'a': 1, 'b': 2}
当调用func(1,2)时，打印：(1,2) {}

> 2 - def _buildNetwork(*layers, **options): 中“_”与“__”的意义。

**_单下划线开头**：弱“内部使用”标识，如：”from M import *”，将不导入所有以下划线开头的对象，包括包、模块、成员

**单下划线结尾_**：只是为了避免与python关键字的命名冲突
 
**__双下划线开头**：模块内的成员，表示私有成员，外部无法直接调用

**双下划线开头双下划线结尾**：指那些包含在用户无法控制的命名空间中的“魔术”对象或属性，如类成员的name、doc、init、import、file、等。推荐永远不要将这样的命名方式应用于自己的变量或函数。


# 步骤

> **S-1 Bind the right class to the Network name**

    Network = network_map[opt['recurrent'], opt['fast']]

> **S-2 Init network** 

    n = Network()
    # linear input layer
    n.addInputModule(LinearLayer(layers[0], name='in'))
    # output layer of type 'outclass'
    n.addOutputModule(opt\['outclass'](layers[-1], name='out'))
    # arbitrary number of hidden layers of type 'hiddenclass'
    for i, num in enumerate(layers[1:-1]):
        layername = 'hidden%i' % i
        n.addModule(opt\['hiddenclass'](num, name=layername))

> **S-3 Connections among layers** 

    # connections between hidden layers
    for i in range(len(layers) - 3):
        n.addConnection(FullConnection(n['hidden%i' % i], n['hidden%i' % (i + 1)]))
    # other connections
    if len(layers) == 2:
        # flat network, connection from in to out
        n.addConnection(FullConnection(n['in'], n['out']))
    else:
        # network with hidden layer(s), connections from in to first hidden and last hidden to out
        n.addConnection(FullConnection(n['in'], n['hidden0']))
        n.addConnection(FullConnection(n['hidden%i' % (len(layers) - 3)], n['out']))

> **S-4 Recurrent connections** 

    # recurrent connections
    if issubclass(opt['hiddenclass'], LSTMLayer):
        if len(layers) > 3:
            errorexit("LSTM networks with > 1 hidden layers are not supported!")
        n.addRecurrentConnection(FullConnection(n['hidden0'], n['hidden0']))

# 结语

PyBrain是Python实现人工神经网络的一个第三方库，可以利用其快速构建神经网络，本次只是展开其构建神经网络的大体步骤，接下来会对具体实现细节进行详细描述。


如果有相关问题，联系邮箱：misterrxzh@gmail.com，或直接[联系我][联系我]留言

[联系我]:  https://mr-rxz.github.io/contact.html
