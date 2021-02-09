import logging
import timeit
import time
import datetime
from logutils.queue import QueueListener, QueueHandler
from queue import Queue
import threading

tmpq = Queue()

def std_manual_threading():
    start = datetime.datetime.now()
    logger = logging.getLogger()
    hdlr = logging.FileHandler('std_manual.out', 'w')
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    def logger_thread(f):
        while True:
            item = tmpq.get(0.1)
            if item == None:
                break
            logging.info(item)
    f = open('manual.out', 'w')
    lt = threading.Thread(target=logger_thread, args=(f,))
    lt.start()
    for i in range(100000):
        tmpq.put("msg:%d" % i)
    tmpq.put(None)
    lt.join()
    print(datetime.datetime.now() - start)

def nonstd_manual_threading():
    start = datetime.datetime.now()
    def logger_thread(f):
        while True:
            item = tmpq.get(0.1)
            if item == None:
                break
            f.write(item+"\n")
    f = open('manual.out', 'w')
    lt = threading.Thread(target=logger_thread, args=(f,))
    lt.start()
    for i in range(100000):
        tmpq.put("msg:%d" % i)
    tmpq.put(None)
    lt.join()
    print(datetime.datetime.now() - start)


def std_logging_queue_handler():
    start = datetime.datetime.now()
    q = Queue(-1)

    logger = logging.getLogger()
    hdlr = logging.FileHandler('qtest.out', 'w')
    ql = QueueListener(q, hdlr)


    # Create log and set handler to queue handle
    root = logging.getLogger()
    root.setLevel(logging.DEBUG) # Log level = DEBUG
    qh = QueueHandler(q)
    root.addHandler(qh)

    ql.start()

    for i in range(100000):
        logging.info("msg:%d" % i)
    ql.stop()
    print(datetime.datetime.now() - start)

def std_logging_single_thread():
    start = datetime.datetime.now()
    logger = logging.getLogger()
    hdlr = logging.FileHandler('test.out', 'w')
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)
    for i in range(100000):
        logging.info("msg:%d" % i)
    print(datetime.datetime.now() - start)

if __name__ == "__main__":
    """
    Conclusion: std logging about 3 times slower so for 100K lines simple file write is ~1 sec while std
    logging ~3. If threads are introduced some overhead causes to go to ~4 and if QueueListener and events
    are used with enhancement for thread sleeping that goes to ~5 (probably because log records are being
    inserted into queue).
    """
    print("Testing")
    std_logging_single_thread() # 3.4
    #std_logging_queue_handler() # 7, 6, 7 (5 seconds with sleep optimization)
    #nonstd_manual_threading() # 1.08
    #std_manual_threading() # 4.3

