import re


token_split_re = re.compile('[ \n]|(\w+)')


def get_line_tokens(conf, line):
    # TODO: endless comment
    it = iter(re.split(conf.token_split_re, line))

    def get_raw_groups(it):
        state = False
        while True:
            try:
                yield (next(it), state)
                state = not state
            except:
                break

    for group, is_word in get_raw_groups(it):
        if not group or (not is_word and group in conf.ignored_tokens):
            continue
        yield (group, is_word)


def get_tokens(conf, lines):
    for line in lines:
        yield from get_line_tokens(conf, line)
        yield ('\n', False)


def group_tokens(conf, tokens):
    curgroup = []
    for token in tokens:
        curgroup.append(token)
        if not token[1] and token[0] in conf.breaking_tokens:
            yield curgroup
            curgroup = []
    if curgroup:
        yield curgroup
