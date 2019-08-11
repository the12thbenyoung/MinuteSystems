from dataMatrixDecoder import process_rack
from multiprocessing import Pool, Process, Queue
from time import sleep

paths = ['image.jpg', 'shpongus.jpg']
process_list = []
data_queue = Queue()

for path in paths:
    proc = Process(target=process_rack, args=(path,data_queue))
    proc.start()
    process_list.append(proc)

for proc in process_list:
    proc.join()

while not data_queue.empty():
    filename, dataIndices, tubesFound, matricesDecoded = data_queue.get()
    print(filename)
    print(dataIndices)
    print(f'tubes found: {tubesFound}')
    print(f'decoded: {matricesDecoded}')
