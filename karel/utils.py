from pyparsing import nestedExpr

def beautifier(inputs, indent=1, tabspace=2):
    lines, queue = [], []
    space = tabspace * " "

    for item in inputs:
        if item == ";":
            lines.append(" ".join(queue))
            queue = []
        elif type(item) == str:
            queue.append(item)
        else:
            lines.append(" ".join(queue + ["{"]))
            queue = []

            inner_lines = beautifier(item, indent=indent+1, tabspace=tabspace)
            lines.extend([space + line for line in inner_lines[:-1]])
            lines.append(inner_lines[-1])

    if len(queue) > 0:
        lines.append(" ".join(queue))

    return lines + ["}"]

def pprint(code, tabspace=2):
    array = nestedExpr('{','}').parseString("{"+code+"}").asList()
    lines = beautifier(array[0])
    print("\n".join(lines[:-1]).replace(' ( ', '(').replace(' )', ')'))
