import ply.lex as lex

tokens = [
        'DEF', 'RUN', 
        'LPAREN', 'RPAREN', 'COLON', 'SEMI', 'INT', #'NEWLINE',
        'WHILE', 'REPEAT',
        'IF', 'IFELSE', 'ELSE',
        'FRONTISCLEAR', 'LEFTISCLEAR', 'RIGHTISCLEAR',
        'MARKERSPRESENT', 'NOMARKERSPRESENT', 'NOT',
        'MOVE', 'TURNRIGHT', 'TURNLEFT',
        'PICKMARKER', 'PUTMARKER',
]

# Tokens
t_ignore =' \t'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COLON  = r':'
t_SEMI   = r';'

t_DEF = 'def'
t_RUN = 'run'
t_WHILE = 'while'
t_REPEAT = 'repeat'
t_IF = 'if'
t_IFELSE = 'ifelse'
t_ELSE = 'else'
t_FRONTISCLEAR = 'frontIsClear'
t_LEFTISCLEAR = 'leftIsClear'
t_RIGHTISCLEAR = 'rightIsClear'
t_MARKERSPRESENT = 'markersPresent'
t_NOMARKERSPRESENT = 'noMarkersPresent'
t_NOT = 'not'
t_MOVE = 'move'
t_TURNRIGHT = 'turnRight'
t_TURNLEFT = 'turnLeft'
t_PICKMARKER = 'pickMarker'
t_PUTMARKER = 'putMarker'

def t_INT(t):
    r'\d+'
    t.value = int(t.value)
    return t

def t_error(t):
    print("Illegal character %s" % repr(t.value[0]))
    t.lexer.skip(1)


lexer = lex.lex() 
if __name__ == '__main__':
    lex.runmain(lexer)

    example = """
        def run():
            repeat(4):
                putMarker()
                move()
                turnLeft()
    """
    lex.input(example)
    while True:
        tok = lex.token()
        if not tok: break
        print(tok)
