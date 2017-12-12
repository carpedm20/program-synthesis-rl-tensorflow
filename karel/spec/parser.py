import ply.yacc as yacc
import lexer

tokens = lexer.tokens

def p_prog(p):
    '''prog : DEF RUN stmt'''

def p_stmt(p):
    '''stmt : WHILE LPAREN cond RPAREN COLON stmt
            | REPEAT LPAREN cste RPAREN COLON stmt
            | stmt SEMI stmt
            | action
            | IF LPAREN cond RPAREN COLON stmt
            | IFELSE LPAREN cond RPAREN COLON stmt ELSE stmt
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
    '''cond : MOVE LPAREN RPAREN
            | TURNRIGHT LPAREN RPAREN
            | TURNLEFT LPAREN RPAREN
            | PICKMARKER LPAREN RPAREN
            | PUTMARKER LPAREN RPAREN
    '''

def p_cste(p):
    '''cste : NUM0
            | NUM1
            | NUM2
            | NUM3
            | NUM4
            | NUM5
            | NUM6
            | NUM7
            | NUM8
            | NUM9
            | NUM10
            | NUM11
            | NUM12
            | NUM13
            | NUM14
            | NUM15
            | NUM16
            | NUM17
            | NUM18
            | NUM19
    '''

yacc.yacc()
if __name__ == '__main__':
    example = """
        def run():
            repeat(4):
                putMarker()
                move()
                turnLeft()
    """
    yacc.parse(example)
