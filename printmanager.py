total_length = 80
all_blanks = '\b' * total_length

statuslens = []

sleep_time = 0.1

UPDATELN = 0
STATUS_ADD = 1
STATUS_UPDATE = 2
STATUS_POP = 3
PRINTLN = 4

def _updateln(*args, **kwargs):
    string = ''
    for arg in args:
        string += str(arg)
    return string

def _status_add(stat, **kwargs):
    string = kwargs['string']
    string += str(stat)
    statuslens.append(len(str(stat)))
    return string

def _status_update(stat, **kwargs):
    stri = _status_pop(string=kwargs['string'])
    return _status_add(stat, string=stri)

def _println(*args, **kwargs):
    stri = ''.join(map(str, args))
    stri += '\n'
    return stri

def _status_pop(*args, **kwargs):
    lastlen = statuslens.pop()
    return kwargs['string'][:-lastlen]

def print_executor():
    global PrintQueue
    functions = [_updateln, _status_add, _status_update, _status_pop, _println]
    stri = ''
    line_length = 80
    while True:
        while True:
            f, args = PrintQueue.get()
            stri = functions[f](*args, string=stri)
            if PrintQueue.empty():
                break
        line_clear = '\b' * line_length + ' ' * line_length + '\b' * line_length
        print(line_clear + stri, end='', flush=True)
        line_length = len(stri) # update based on how much is printed
        FlushEvent.set()
        PrintEvent.wait(sleep_time)
        # clear the flag if set
        PrintEvent.clear()

use_multiprocessing = True

if use_multiprocessing:
    try:
        from multiprocessing import Process, Queue, Event
        from time import sleep
        PrintProcess = Process(target=print_executor)
        PrintQueue = Queue()
        PrintEvent = Event()
        FlushEvent = Event()
        PrintProcess.daemon = True
        PrintProcess.start()
    except:
        use_multiprocessing = False

def handle_func(func, num, args, immediate):
    if not use_multiprocessing:
        func(*args)
    else:
        PrintQueue.put_nowait((num, args))
        if immediate:
            PrintEvent.set()
            FlushEvent.wait(sleep_time)
            FlushEvent.clear()

def updateln(*args, flush=False):
    global _updateln
    handle_func(_updateln, UPDATELN, args, flush)

def status_add(*args, flush=False):
    global _status_add
    handle_func(_status_add, STATUS_ADD, args, flush)

def status_update(*args, flush=False):
    global _status_update
    handle_func(_status_update, STATUS_UPDATE, args, flush)

def status_pop(*args, flush=False):
    global _status_pop
    handle_func(_status_pop, STATUS_POP, args, flush)

def println(*args, flush=True):
    global _println
    handle_func(_println, PRINTLN, args, flush)
