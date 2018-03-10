import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter


# layer parameters corresponding to several data
layer_params = {}
layer_params['mnist'] = [32, 32, 64, 64, 200, 200]
layer_params['cifar10'] = [64, 64, 128, 128, 256, 256]
layer_params['fmnist'] = [64, 64, 128, 128, 256, 256]


# Nueral Net classifer
class ClassiferNN(chainer.Chain):
    
    def __init__(self, layer_params, T=1.0, n_classes=10):
        super(ClassiferNN, self).__init__()
        
        self.T = T # Temperature
        self.n_classes = n_classes
        
        with self.init_scope():
            self.c1 = L.Convolution2D(None, layer_params[0], ksize=3)
            self.c2 = L.Convolution2D(None, layer_params[1], ksize=3)
            self.c3 = L.Convolution2D(None, layer_params[2], ksize=3)
            self.c4 = L.Convolution2D(None, layer_params[3], ksize=3)
            
            self.l5 = L.Linear(None, layer_params[4])
            self.l6 = L.Linear(None, layer_params[5])
            self.l7 = L.Linear(None, n_classes)
            
            
    # forward propagation
    def __call__(self, x):
        h1 = F.relu(self.c1(x))
        h2 = F.relu(self.c2(h1))
        h3 = F.max_pooling_2d(h2, ksize=2)
        
        h4 = F.relu(self.c3(h3))
        h5 = F.relu(self.c4(h4))
        h6 = F.max_pooling_2d(h5, ksize=2)
        
        h7 = F.dropout(F.relu(self.l5(h6)), 0.5)
        h8 = F.relu(self.l6(h7))
        h9 = self.l7(h8) / self.T
        
        return h9
    

    def predict_proba(self, x):
        return F.softmax(self.__call__(x))
    
    
    def cross_entropy(self, x, t):
        y = self.__call__(x)
        return F.softmax_cross_entropy(y, t)
    
    
    def accuracy(self, x, t):
        y = self.__call__(x)
        return F.accuracy(y, t)
    
    
    # loss as hard labeling cross entropy
    def hard_cross_entropy_loss(self, x, t):
        loss = self.cross_entropy(x, t)
        accu = self.accuracy(x, t)
        # reporting loss and accuracy
        reporter.report({'loss':loss, 'accuracy':accu}, self)
        return loss
    
    
    # loss as a soft labeling cross entropy used when distillation
    def soft_cross_entropy_loss(self, x, p):
        # in the argment of log(), add +1 to care numerical instability
        loss = -F.mean(F.sum(p * F.log(1 + self.predict_proba(x)), axis=1))
        # reporting loss
        reporter.report({'loss':loss}, self)
        return loss
