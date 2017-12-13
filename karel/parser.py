import yacc
import numpy as np
import ply.lex as lex
from pyparsing import nestedExpr

from __init__ import Karel

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
    print("\n".join(lines[:-1]))

class Parser(object):
    """
    Base class for a lexer/parser that has the rules defined as methods
    """
    tokens = ()
    precedence = ()

    def __init__(self, rng=None, **kwargs):
        self.names = {}
        self.debug = kwargs.get('debug', 0)

        if rng is None:
            self.rng = np.random.RandomState(1)
        else:
            self.rng = rng

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

    #########
    # lexer
    #########

    MIN_INT = 0
    MAX_INT = 19

    def t_INT(self, t):
        r'\d+'

        value = int(t.value)
        if not (self.MIN_INT <= value <= self.MAX_INT):
            raise Exception(" [!] Out of range ({} ~ {}): `{}`". \
                    format(self.MIN_INT, self.MAX_INT, value))

        t.value = value
        return t

    def random_INT(self):
        return self.rng.randint(self.MIN_INT, self.MAX_INT + 1)

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

    def random_generate(self, start_token="prog", depth=0, stmt_max_depth=3):
        if start_token == 'stmt' and depth > stmt_max_depth:
            start_token = "action"

        codes = []
        candidates = self.prodnames[start_token]

        prod = candidates[self.rng.randint(len(candidates))]

        for term in prod.prod:
            if term in self.prodnames: # need digging
                codes.extend(self.random_generate(term, depth + 1, stmt_max_depth))
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

    code = """def run(): repeat(2): turn_left(); move();"""
    print("RUN:", parser.run(code))

    parser.draw()

    for idx in range(100):
        print("="*10)
        pprint(" ".join(parser.random_generate()))
