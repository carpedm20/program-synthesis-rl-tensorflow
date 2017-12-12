import ply.yacc as yacc
import lexer

tokens = lexer.tokens

def p_prog(p):
    '''prog : DEF RUN LPAREN RPAREN COLON stmt'''

def p_stmt(p):
    '''stmt : WHILE LPAREN RPAREN COLON stmt
            | REPEAT LPAREN cste RPAREN COLON stmt
            | stmt SEMI stmt
            | stmt NEWLINE stmt
            | action
            | IF LPAREN cond RPAREN COLON stmt
            | IFELSE LPAREN cond RPAREN COLON stmt ELSE stmt
            | empty
    '''

def p_cond(p):
    '''cond : FRONTISCLEAR LPAREN RPAREN
            | LEFTISCLEAR LPAREN RPAREN
            | RIGHTISCLEAR LPAREN RPAREN
            | MARKERSPRESENT LPAREN RPAREN 
            | NOMARKERSPRESENT LPAREN RPAREN
            | NOT cond
    '''

def p_action(p):
    '''action : MOVE LPAREN RPAREN
              | TURNRIGHT LPAREN RPAREN
              | TURNLEFT LPAREN RPAREN
              | PICKMARKER LPAREN RPAREN
              | PUTMARKER LPAREN RPAREN
    '''

def p_cste(p):
    '''cste : INT
    '''

def p_empty(p):
    """empty :"""
    pass

def p_error(p):
    if p:
        print("Syntax error at '%s'" % p.value)
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()
if __name__ == '__main__':
    example = """
        def run():
            repeat(4):
                putMarker()
                move();
                turnLeft()
    """
    parser.parse(example, debug=True)
