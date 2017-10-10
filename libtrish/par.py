import os


def par_list_map(f, l, lsize, proc):
    plist = []
    for i in range(proc):
        pid = os.fork()
        if pid != 0:
            plist.append(pid)
            continue
        for i in range(i, lsize, proc):
            f(l[i])
        return
    for pid in plist:
        os.waitpid(pid, 0)
