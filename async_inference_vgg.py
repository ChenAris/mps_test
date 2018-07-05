import mxnet as mx
import cv2
import time
import numpy as np
import sys
from threading import Thread
ctx = mx.gpu(0)
batch_size = int(sys.argv[2])
sym, arg_params, aux_params = mx.model.load_checkpoint('vgg16', 0)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (batch_size,3,224,224))], 
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=True)
with open('synset.txt', 'r') as f:
    labels = [l.rstrip() for l in f]
    
global_count = 0
#imgs = get_images(1000)
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

class PredictThread(Thread):
    def __init__(self,latency_list,idx):
	Thread.__init__(self)
        self.latency_list = latency_list
        self.idx = idx
    
    def run(self):
        self.latency_list = async_inference(self.latency_list,self.idx)
#	print "Thread %s, finish time:" %(self.getName()),time.time()
#	global global_count
#	global_count += 1 
#	print "finish forward times:",global_count
#	print self.latency_list

    def get_result(self):
#	print "Thread %s is_alive:" %(self.getName()),self.is_alive()
	self.join()
#	print "Thread %s is_alive:" %(self.getName()),self.is_alive()
#	print "Trying to get result"
#	while self.is_alive():		
        return self.latency_list
        
def get_image1(fname, show=False):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
         return None
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img

def get_images(num):
    idx = range(num)
    imgs = []
    for i in idx:
	imgs.append(get_image1('data/val_1000/%d.jpg' %(i)))
    return imgs

imgs = get_images(1000)
def predict(num):
    tic_total = time.time()
    latency = [0 for i in range(0,batch_size)]
    latency_count = 0   
    batch_count = 0 
    tic_get_imgs = time.time()
#    imgs = get_images(1000)
    tic_latency_count = 0
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
#	    tic_start = time.time()
            img = np.concatenate([imgs[j] for j in idx])
            mod.forward(Batch([mx.nd.array(img)]))
            prob = mod.get_outputs()[0].asnumpy()
            pred = np.argsort(prob,axis = 1)
            top1 = pred[:,-1]
            latency=[time.time()-latency[j] for j in range(0,batch_size)]
#	    print "Sync time:",time.time() - tic_start
	    tic_latency = time.time()
            for j in range(0,batch_size):
                latency_count += latency[j]
	    tic_latency_count += time.time() - tic_latency

    print "total latency time:",tic_latency_count
    inference_time = time.time() - tic_total
    throughput = num/float(inference_time)
    print "inference_time:",inference_time," num:",num

    throughput = num/float(inference_time)
    print 'VGG Throughput:',throughput,' Average Latency:',float(latency_count)/num

def async_inference(latency_list,idx):
    img = np.concatenate([imgs[j] for j in idx])
    mod.forward(Batch([mx.nd.array(img)]))
    prob = mod.get_outputs()[0].asnumpy()
    pred = np.argsort(prob,axis = 1)
    top1 = pred[:,-1]
 #   print top1
    latency=[time.time()-latency_list[j] for j in range(0,batch_size)]
    return latency

def async_predict(num):
    tic_total = time.time()
    latency = [0 for i in range(0,batch_size)]
    latency_count = 0   
    batch_count = 0 
    tic_get_imgs = time.time()
    predict_thread = []
    tic_first = time.time()
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
            #latency_finish = async_inference(latency,idx)
	    th = PredictThread(latency,idx)
	    tic_start = time.time()
	    th.start()
	#    print "For async, start time:",time.time() - tic_start
            predict_thread.append(th)
 #           for j in range(0,batch_size):
 #               latency_count += latency_finish[j]

    toc_last = time.time()
    for i in range(len(predict_thread)):
	#print "Getting result for thread ",i
        latency_temp = predict_thread[i].get_result()
        for j in range(len(latency_temp)):
            latency_count += latency_temp[j]

    inference_time = time.time() - tic_total
#    print "Finish async inference time:",toc_last-tic_first," Total time:",inference_time," latency time:",time.time()-toc_last
    throughput = num/float(inference_time)
#    print "inference_time:",inference_time," num:",num
    print 'VGG Throughput:',throughput,' Average Latency:',float(latency_count)/num    

import datetime
def main():
    start = time.clock()
    start_date =  datetime.datetime.now()
    num = sys.argv[1]
    num = int(num)
    tic_sync = time.time()
   # predict(num)
 #   print "Time for Sync:",time.time() - tic_sync
    tic_async = time.time()
    async_predict(num)
  #  print "Time for Async:",time.time() - tic_async
    elapsed = time.clock() - start
    end_date =  datetime.datetime.now()-start_date

    

if __name__ == "__main__":
    main()
