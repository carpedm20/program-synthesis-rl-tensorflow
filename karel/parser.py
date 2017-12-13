import yacc
import numpy as np
import ply.lex as lex

from utils import pprint
from __init__ import Karel

class Parser(object):
    """
    Base class for a lexer/parser that has the rules defined as methods
    """
    tokens = ()
    precedence = ()

    def __init__(self, **kwargs):
        self.names = {}
        self.debug = kwargs.get('debug', 0)

        # Build the lexer and parser
        lex.lex(module=self, debug=self.debug)
        _, self.grammar = yacc.yacc(
                module=self, debug=self.debug,
                tabmodule='_parsetab', with_grammar=True)

        self.prodnames = self.grammar.Prodnames

class KarelParser(Parser):

    tokens = [
            'DEF', 'RUN', 
            'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'SEMI', 'INT', #'NEWLINE',
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
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_SEMI   = r';'

    t_DEF = 'def'
    t_RUN = 'run'
    t_WHILE = 'while'
    t_REPEAT = 'repeat'
    t_IF = 'if'
    t_IFELSE = 'ifelse'
    t_ELSE = 'else'
    t_NOT = 'not'

    t_FRONTISCLEAR = 'front_is_clear'
    t_LEFTISCLEAR = 'left_is_clear'
    t_RIGHTISCLEAR = 'right_is_clear'
    t_MARKERSPRESENT = 'markers_present'
    t_NOMARKERSPRESENT = 'no_markers_present'

    t_MOVE = 'move'
    t_TURNRIGHT = 'turn_right'
    t_TURNLEFT = 'turn_left'
    t_PICKMARKER = 'pick_marker'
    t_PUTMARKER = 'put_marker'


    def __init__(self, rng=None, min_int=0, max_int=19, **kwargs):
        super(KarelParser, self).__init__(**kwargs)

        self.min_int = min_int
        self.max_int = max_int

        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng


    #########
    # lexer
    #########

    def t_INT(self, t):
        r'\d+'

        value = int(t.value)
        if not (self.min_int <= value <= self.max_int):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`". \
                    format(self.min_int, self.max_int, value))

        t.value = value
        return t

    def random_INT(self):
        return self.rng.randint(self.min_int, self.max_int + 1)

    def t_error(self, t):
        print("Illegal character %s" % repr(t.value[0]))
        t.lexer.skip(1)

    #########
    # parser
    #########

    def p_prog(self, p):
        '''prog : DEF RUN LPAREN RPAREN LBRACE stmt RBRACE'''

    def p_stmt(self, p):
        '''stmt : WHILE LPAREN cond RPAREN LBRACE stmt RBRACE
                | REPEAT LPAREN cste RPAREN LBRACE stmt RBRACE
                | stmt SEMI stmt
                | action
                | IF LPAREN cond RPAREN LBRACE stmt RBRACE
                | IFELSE LPAREN cond RPAREN LBRACE stmt RBRACE ELSE LBRACE stmt RBRACE
        '''
        return p

    def p_cond(self, p):
        '''cond : cond_without_not
                | NOT cond_without_not
        '''

    def p_cond_without_not(self, p):
        '''cond_without_not : FRONTISCLEAR LPAREN RPAREN
                            | LEFTISCLEAR LPAREN RPAREN
                            | RIGHTISCLEAR LPAREN RPAREN
                            | MARKERSPRESENT LPAREN RPAREN 
                            | NOMARKERSPRESENT LPAREN RPAREN
        '''

    def p_action(self, p):
        '''action : MOVE LPAREN RPAREN
                | TURNRIGHT LPAREN RPAREN
                | TURNLEFT LPAREN RPAREN
                | PICKMARKER LPAREN RPAREN
                | PUTMARKER LPAREN RPAREN
        '''
        #fn_name = p[1]
        #if fn_name is not None:
        #    getattr(self.karel, fn_name)()

        return p

    def p_cste(self, p):
        '''cste : INT
        '''
        return int(p[1])

    def p_empty(self, p):
        """empty :"""

    def p_error(self, p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")

    #########
    # Karel
    #########

    def run(self, code, **kwargs):
        return yacc.parse(code, **kwargs)

    def new_game(self, **kwargs):
        self.karel = Karel(**kwargs)

    def draw(self, **kwargs):
        self.karel.draw(**kwargs)

    def random_code(self, **kwargs):
        return " ".join(self.random_tokens(**kwargs))

    def random_tokens(self, start_token="prog", depth=0, stmt_max_depth=3):
        if start_token == 'stmt' and depth > stmt_max_depth:
            start_token = "action"

        codes = []
        candidates = self.prodnames[start_token]

        prod = candidates[self.rng.randint(len(candidates))]

        for term in prod.prod:
            if term in self.prodnames: # need digging
                codes.extend(self.random_tokens(term, depth + 1, stmt_max_depth))
            else:
                token = getattr(self, 't_{}'.format(term))
                if callable(token):
                    if token == self.t_INT:
                        token = self.random_INT()
                    else:
                        raise Exception(" [!] Undefined token `{}`".format(token))

                codes.append(str(token).replace('\\', ''))

        return codes


if __name__ == '__main__':
    parser = KarelParser()

    parser.new_game(world_size=(4, 4))
    parser.draw()

    code = """def run ( ) { ifelse ( not right_is_clear ( ) ) { repeat ( 4 ) { move ( ) } } else { ifelse ( not right_is_clear ( ) ) { pick_marker ( ) ; put_marker ( ) } else { put_marker ( ) } } }"""
    print("RUN:", parser.run(code))

    parser.draw()

    for idx in range(5):
        print("="*10)
        code = parser.random_code()
        print(code)
