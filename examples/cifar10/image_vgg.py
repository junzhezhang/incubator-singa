from singa.layer import Conv2D, Activation, MaxPooling2D, \
        Flatten, Dense, BatchNormalization, Dropout
from singa import net as ffnet
from singa import initializer
from singa import layer


ffnet.verbose=False
cfg = {
        11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def create_layers(net, cfg, sample_shape, batch_norm=False):
    lid = 0
    for idx, v in enumerate(cfg):
        if v == 'M':
            net.add(MaxPooling2D('pool/features.%d' % lid, 2, 2, pad=0))
            lid += 1
        else:
            net.add(Conv2D('conv/features.%d' % lid, v, 3, pad=1,
                           input_sample_shape=sample_shape))
            lid += 1
            if batch_norm:
                net.add(BatchNormalization('bn/features.%d' % lid))
                lid += 1
            net.add(Activation('act/features.%d' % lid))
            lid += 1
        sample_shape = None
    return net


def init_params(net, weight_path=None):
    if weight_path is None:
        print ("======== to init param")
        for pname, pval in zip(net.param_names(), net.param_values()):
            print(pname, pval.shape,len(pval.shape))
            if len(pval.shape) == 2:
              initializer.uniform(x, pval.shape[0], pval.shape[1])
            elif len(pval.shape) == 1:
              initializer.uniform(x, pval.shape[0])
              print ("condition shape 1")
            else:
              print ("not in any condition, DSB!!!")
            # if 'conv' in pname and len(pval.shape) > 1:
            #     initializer.gaussian(pval, 0, pval.shape[1])
            #     print ('conv')
            #     print (pval.shape)
            #     print (pval.shape[1])
            # elif 'dense' in pname:
            #     if len(pval.shape) > 1:
            #         initializer.gaussian(pval, 0, pval.shape[0])
            #     else:
            #         pval.set_value(0)
            # # init params from batch norm layer
            # elif 'mean' in pname or 'beta' in pname:
            #     pval.set_value(0)
            # elif 'var' in pname or 'gamma' in pname:
            #     pval.set_value(1)
            # else:
            #   print ("not in any condition, DSB!!!")
              # print (pval.shape)
              # print (pval.shape[0])
              # print (pname.shape)
              # initializer.gaussian(pval, 0, pval.shape[0])
              
    else:
        net.load(weight_path, use_pickle='pickle' in weight_path)


def create_net(depth, nb_classes, batchnorm=False, use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'
    net = ffnet.FeedForwardNet()
    net = create_layers(net, cfg[depth], (3, 224, 224), batchnorm)
    net.add(Flatten('flat'))
    net.add(Dense('dense/classifier.0', 4096))
    net.add(Activation('act/classifier.1'))
    net.add(Dropout('dropout/classifier.2'))
    net.add(Dense('dense/classifier.3', 4096))
    net.add(Activation('act/classifier.4'))
    net.add(Dropout('dropout/classifier.5'))
    net.add(Dense('dense/classifier.6', nb_classes))
    return net

if __name__ == '__main__':
    create_net(13, 1000)