import ply.lex as lex

tokens = [
        'ID', 'LPAREN', 'RPAREN', 'COLON', 'SEMI', 'INT',
        #'IF', 'IFELSE', 'ELSE',
        #'FRONTISCLEAR', 'LEFTISCLEAR', 'RIGHTISCLEAR',
        #'MARKERSPRESENT', 'NOMARKERSPRESENT',
        #'MOVE', 'TURNRIGHT', 'TURNLEFT',
        #'PICKMARKER', 'PUTMARKER',
]

# Reserved words
reserved = {
    'def': 'DEF',
    'run': 'RUN',
    'while': 'WHILE',
    'repeat': 'REPEAT',
    'if': 'IF',
    'ifelse': 'IFELSE',
    'else': 'ELSE',
    'frontIsClear': 'FRONTISCLEAR',
    'leftIsClear': 'LEFTISCLEAR',
    'rightIsClear': 'RIGHTISCLEAR',
    'markersPresent': 'MARKERSPRESENT',
    'noMarkersPresent': 'NOMARKERSPRESENT',
    'not': 'NOT',
    'move': 'MOVE',
    'turnRight': 'TURNRIGHT',
    'turnLeft': 'TURNLEFT',
    'pickMarker': 'PICKMARKER',
    'putMarker': 'PUTMARKER',
}
tokens += reserved.values()

# Tokens
t_ignore = ' \t\x0c'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COLON  = r':'
t_SEMI   = r';'

def t_ID(t):
    r'[A-Za-z_][\w_]*'
    t.type = reserved.get(t.value, "ID")
    return t

def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_error(t):
    print("Illegal character %s" % repr(t.value[0]))
    t.lexer.skip(1)


lex.lex() 
if __name__ == '__main__':
    example_code = """
    def run():
        repeat(4):
            putMarker()
            move()
            turnLeft()
    """
    lex.input(example_code)
    while True:
        tok = lex.token()
        if not tok: break
        print(tok)
