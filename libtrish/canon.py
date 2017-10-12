def canonicalize(conf, groups):
    def canonicalize_sub(group):
        wmap = {}
        wlist = []
        i = 0
        for group, is_word in group:
            if not is_word:
                wlist.append(group)
                continue
            gstatus = wmap.get(group)
            if gstatus is not None:
                wlist.append(gstatus)
            else:
                wmap[group] = i
                wlist.append(i)
                i += 1
        return tuple(wlist)
    return map(canonicalize_sub, groups)
