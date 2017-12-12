import yacc as yacc
import ply.lex as lex


class Parser(object):
    """
    Base class for a lexer/parser that has the rules defined as methods
    """
    tokens = ()
    precedence = ()

    def __init__(self, **kw):
        self.debug = kw.get('debug', 0)
        self.names = {}
        try:
            modname = os.path.split(os.path.splitext(__file__)[0])[
                1] + "_" + self.__class__.__name__
        except:
            modname = "parser" + "_" + self.__class__.__name__
        self.debugfile = modname + ".dbg"
        self.tabmodule = modname + "_" + "parsetab"
        # print self.debugfile, self.tabmodule

        # Build the lexer and parser
        lex.lex(module=self, debug=self.debug)
        yacc.yacc(module=self,
                  debug=self.debug,
                  debugfile=self.debugfile,
                  tabmodule=self.tabmodule)

    def run(self, code, **kwargs):
        return yacc.parse(code, **kwargs)

class KarelParser(Parser):

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

    def t_INT(self, t):
        r'\d+'
        t.value = int(t.value)
        return t

    def t_error(self, t):
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)

    def p_prog(self, p):
        '''prog : DEF RUN LPAREN RPAREN COLON stmt'''

    def p_stmt(self, p):
        '''stmt : WHILE LPAREN RPAREN COLON stmt
                | REPEAT LPAREN cste RPAREN COLON stmt
                | stmt SEMI stmt
                | action
                | IF LPAREN cond RPAREN COLON stmt
                | IFELSE LPAREN cond RPAREN COLON stmt ELSE stmt
        '''

    def p_cond(self, p):
        '''cond : FRONTISCLEAR LPAREN RPAREN
                | LEFTISCLEAR LPAREN RPAREN
                | RIGHTISCLEAR LPAREN RPAREN
                | MARKERSPRESENT LPAREN RPAREN 
                | NOMARKERSPRESENT LPAREN RPAREN
                | NOT cond
        '''

    def p_action(self, p):
        '''action : MOVE LPAREN RPAREN
                | TURNRIGHT LPAREN RPAREN
                | TURNLEFT LPAREN RPAREN
                | PICKMARKER LPAREN RPAREN
                | PUTMARKER LPAREN RPAREN
                | empty
        '''

    def p_cste(self, p):
        '''cste : INT
        '''

    def p_empty(self, p):
        """empty :"""

    def p_error(self, p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")


if __name__ == '__main__':
    parser = KarelParser()

    code = """def run(): move();"""
    print(parser.run(code))
