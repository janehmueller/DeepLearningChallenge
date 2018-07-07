import threading


# taken form https://stackoverflow.com/questions/1131430/are-generators-threadsafe
class LockedIterator(object):
    def __init__(self, it):
        self.lock = threading.Lock()
        self.it = it.__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        self.lock.acquire()
        try:
            return self.it.__next__()
        finally:
            self.lock.release()
