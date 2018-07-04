import mxnet as mx
import os,sys
import time
import subprocess

batch_size = 32
average_times = 1
process_times = 1
model_name = "resnet"
num = 500

def run_inference(batch_size=32,model_name='vgg',num=100,process_times=1):
    command = "python batch_inference_"+model_name+".py "+str(num)+" "+str(batch_size)
    #output = os.popen(command)
    child_pool = []
    for i in range(process_times):
        child_pool.append(subprocess.Popen(command, stdout=subprocess.PIPE,shell=True))
    record_mark = 0
    end_mark = 1
    gpu_active_count = 0
    total_gpu_uti = 0
    cpu_active_count = 0
    cpu_uti = os.popen("ps ux | grep python| grep inference").readlines() 
    total_cpu_uti = [0 for i in range(len(cpu_uti)-1)]
    memory_gpu = 0       
    while end_mark:
        end_mark = 0
        for j in range(len(child_pool)):
            if child_pool[j].poll() is None:
                end_mark += 1
#        tic = time.time()
        memory_gpu_temp = float(os.popen("nvidia-smi --query-gpu=memory.used --format=csv").readlines()[1].strip('MiB%\n'))
#	print "query nvidia-smi gpu-mempry overhead:",time.time()-tic," so far:",time.time()-tic
        if memory_gpu_temp > memory_gpu:
            memory_gpu = memory_gpu_temp
#	tic_cpu = time.time()
        cpu_uti = os.popen("ps ux | grep python| grep inference|grep -v sh").readlines()
#	print "query cpu overhead:",time.time()-tic_cpu," so far:",time.time()-tic
        cpu_active_count += 1
	try:
            for k in range(len(cpu_uti)-1):
#	    print "Measure CPU:",cpu_uti[k]
                total_cpu_uti[k] += float(cpu_uti[k].split()[2])
	except:
		print cpu_uti
#	tic_gpu_uti = time.time()
        gpu_uti = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv").readlines()
#	toc1 = time.time()
#	gpu_uti2 = os.popen("nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv").readlines()
#	print "query nvidia-smi gpu1 -i method overhead:",time.time()-toc1," without-i method:",toc1-tic_gpu_uti
#	tic_datahandle = time.time()
        gpu_uti = int(gpu_uti[1].strip('%\n'))
#	print time.time() - tic_datahandle
        if gpu_uti > 0 or record_mark:
            total_gpu_uti += gpu_uti
            gpu_active_count += 1
	    record_mark = 1
#        time.sleep(0.1)
#	print "datahanle overhead:",time.time()-tic_datahandle," total overhead:",time.time()-tic

    avg_cpu_uti = [total_cpu_uti[i] / cpu_active_count for i in range(len(total_cpu_uti))]
    avg_gpu_uti = total_gpu_uti / gpu_active_count 
    throughput = []
    avg_latency = []
    for i in range(len(child_pool)):
        line = child_pool[i].communicate()[0].strip('\n').split(' ')
        throughput.append(line[2])
        avg_latency.append(line[6])

    return avg_gpu_uti,throughput,avg_latency,avg_cpu_uti,memory_gpu


def run_main(model_name):
#    batch_sizes = [1,2,4,8,16,24,32,48,64,96,128,196]
#    batch_sizes = [32,2,4,8,16,24,1,48,64,96,128,196]
#    batch_sizes = [16,24,32,48,64,96,128,196]
    batch_sizes = [196,128,96,64,48]
    test_times = 5
    
    for batch_size in batch_sizes:
        os.popen('sudo ./start_as_root.bash')
	if batch_sizes < 96:
	    loop_number = batch_size * 1000
	else:
	    loop_number = batch_size * 500
        #model_name = "resnet-152"
        avg_gpu = []
        avg_throughput = []
        avg_latency = []
        avg_cpu_uti = []
        avg_memory_gpu = 0
        for j in range(test_times):
            gpu_uti,throughput,latency,cpu_uti,memory_gpu = run_inference(batch_size,model_name,loop_number,1)
            avg_gpu.append(gpu_uti)
            avg_throughput.append(throughput)
            avg_latency.append(latency)
            avg_cpu_uti.append(cpu_uti)
            avg_memory_gpu += memory_gpu
            time.sleep(3)

	avg_memory_gpu = avg_memory_gpu / test_times
        fname = open(model_name+'.log','a+')
	fname.write(model_name+" with MPS, batch_size:"+str(batch_size)+" img_num:"+str(loop_number)+" memory used:"+str(avg_memory_gpu)+"\n")
        print model_name+" with MPS, batch_size:"+str(batch_size)+" img_num:"+str(loop_number)+" memory used:"+str(avg_memory_gpu)
        fname.write(str(avg_gpu)+'\n')
        fname.write(str(avg_throughput)+'\n')
        fname.write(str(avg_latency)+'\n')
        fname.write(str(avg_cpu_uti)+'\n')
        fname.close()

        os.popen('sudo ./stop_as_root.bash')

        #model_name = "resnet-152"
        avg_gpu = []
        avg_throughput = []
        avg_latency = []
        avg_cpu_uti = []
        avg_memory_gpu = 0
        for j in range(test_times):
            gpu_uti,throughput,latency,cpu_uti,memory_gpu = run_inference(batch_size,model_name,loop_number,1)
            avg_gpu.append(gpu_uti)
            avg_throughput.append(throughput)
            avg_latency.append(latency)
            avg_cpu_uti.append(cpu_uti)
            avg_memory_gpu += memory_gpu

        fname = open(model_name+'.log','a+')
        fname.write(model_name+" without MPS, batch_size:"+str(batch_size)+" img_num:"+str(loop_number)+" memory used:"+str(avg_memory_gpu)+"\n")
        print model_name+" without MPS, batch_size:"+str(batch_size)+" img_num:"+str(loop_number)+" memory used:"+str(avg_memory_gpu)
        fname.write(str(avg_gpu)+'\n')
        fname.write(str(avg_throughput)+'\n')
        fname.write(str(avg_latency)+'\n')
	fname.write(str(avg_cpu_uti)+'\n')
        fname.close()

      #  batch_size = batch_size*2
def main():
#    model_name = ['resnet','resnet-152','vgg','squee']
    model_name = ['vgg'] 
    for model in model_name:
	os.popen('rm '+model+'.log')
	run_main(model)


if __name__ == "__main__":
    main()
