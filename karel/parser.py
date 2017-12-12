import yacc as yacc
import ply.lex as lex

from __init__ import Karel

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
    t_FRONTISCLEAR = 'front_is_clear'
    t_LEFTISCLEAR = 'left_is_clear'
    t_RIGHTISCLEAR = 'right_is_clear'
    t_MARKERSPRESENT = 'markers_present'
    t_NOMARKERSPRESENT = 'no_markers_present'
    t_NOT = 'not'
    t_MOVE = 'move'
    t_TURNRIGHT = 'turn_right'
    t_TURNLEFT = 'turn_left'
    t_PICKMARKER = 'pick_marker'
    t_PUTMARKER = 'put_marker'

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
        fn_name = p[1]
        if fn_name is not None:
            getattr(self.karel, fn_name)()

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

    def run(self, code, **kwargs):
        return yacc.parse(code, **kwargs)

    def new_game(self, **kargvs):
        self.karel = Karel(**kargvs)

    def draw(self, **kargvs):
        self.karel.draw(**kargvs)

if __name__ == '__main__':
    parser = KarelParser()

    parser.new_game(world_size=(4, 4))
    parser.draw()
    code = """def run(): turn_left(); move();"""

    parser.run(code)
    parser.draw()
