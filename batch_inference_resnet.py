import mxnet as mx
import cv2
import time
#import matplotlib.pyplot as plt
import numpy as np
import sys
#import matplotlib
#matplotlib.use('Agg')
ctx = mx.gpu(0)
batch_size = int(sys.argv[2])
#sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
sym, arg_params, aux_params = mx.model.load_checkpoint('resnet-18', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (batch_size,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]

#print("finish prerequisite")
# define a simple data batch
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

def get_image(url, show=False):
    # download and show the image
    fname = mx.test_utils.download(url)
    img = mx.image.imread(fname)
    if img is None:
        return None
#    if show:
#        plt.imshow(img.asnumpy())
#        plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = mx.image.imresize(img, 224, 224) # resize
    img = img.transpose((2, 0, 1)) # Channel first
    img = img.expand_dims(axis=0) # batchify
    return img

def get_image1(fname, show=False):
#    url = "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/doc/tutorials/python/predict_image/cat.jpg"
#    fname = mx.test_utils.download(url)

    img = cv2.imread(fname)
#    if img == None:
#	raise Exception("could not load image!")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
         return None
#    if show:
#         plt.imshow(img)
#         plt.axis('off')
    # convert into format (batch, RGB, width, height)
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def predict(num):
    tic_total = time.time()
    latency = [0 for i in range(0,batch_size)]
    latency_count = 0
    batch_count = 0
    for i in range(0,num):
        tic_iter = time.time()
        latency[batch_count] = tic_iter
        batch_count += 1
        if (i+1)%batch_size == 0:
            batch_count = 0
            if (i+1)>1000:
                i=i%1000
            if i < batch_size:
                i=batch_size

            idx = range(i+1-batch_size, i+1)
#           print 'current index:',i,' mod:',(i+1)%batch_size
            img = np.concatenate([get_image1('data/val_1000/%d.jpg' %(j)) for j in idx])
            mod.forward(Batch([mx.nd.array(img)]))
            prob = mod.get_outputs()[0].asnumpy()
            pred = np.argsort(prob,axis = 1)
            top1 = pred[:,-1]
            latency=[time.time()-latency[j] for j in range(0,batch_size)]
            for j in range(0,batch_size):
                latency_count += latency[j]

#       print('batch %d, time %f sec' %(i,time.time()-tic_iter))

    inference_time = time.time() - tic_total
    throughput = num/float(inference_time)
 #   print('Total time used for vgg prediction:',time.time()-tic_total)
    print 'Resnet Throughput:',throughput,' Average Latency:',float(latency_count)/num

#    print mod.get_outputs()[0]
#    prob = mod.get_outputs()[0].asnumpy()
#    print ("Get outputs")
    # print the top-5
#    prob = np.squeeze(prob)
#    print ("print the top-5")
#    a = np.argsort(prob)[::-1]
#    for i in a[0:5]:
 #       print('probability=%f, class=%s' %(prob[i], labels[i]))
import datetime
def main():
    start = time.clock()
    start_date =  datetime.datetime.now()
    num = sys.argv[1]
    num = int(num)
    predict(num)
#    for i in range(0,1000):
#	url = 'data/val_1000/%d.jpg' %(i,)
#	print "currently predict img: %s" %url 
#	predict(url)
    elapsed = time.clock() - start
    end_date =  datetime.datetime.now()-start_date
#    print("Time used for vgg16:",elapsed,end_date.seconds)
    

if __name__ == "__main__":
    main()
