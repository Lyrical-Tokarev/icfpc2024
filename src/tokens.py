from dataclasses import dataclass
import numpy as np
import networkx as nx


def wrap_with_token(value):
    if isinstance(value, CommonToken):
        return value
    token = None
    if isinstance(value, str):
        token = StringToken(value)
    elif isinstance(value, bool):
        token = BooleanToken(value)
    elif isinstance(value, int):
        if value < 0:
            token = UnaryToken("-", IntegerToken(-value))
        else:
            token = IntegerToken(value)
            # print("int token creation")
    if token is None:
        print("token is None, value=", value, type(value))
    return token


@dataclass
class CommonToken:
    """Class for tokens"""
    INDICATOR = None
    NUM_PARAMETERS = 0
    def __init__(self, indicator: str, body: str, value=None):
        self.indicator = indicator
        self.body = body
        self.value = value
    
    @classmethod
    def is_match(cls, text):
        # TODO: add condition "(!') up to and including code 126 ('~)"
        return len(text) >= 1 

    @classmethod
    def parse(cls, token_str):
        # indicator = text[0]
        assert cls.is_match(token_str)
        body = token_str[1:]
        return CommonToken(body)

    def __str__(self):
        return f"{self.INDICATOR}{self.body}"

    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}({self.value})"

    @property
    def is_complex(self):
        return self.NUM_PARAMETERS > 0

    def has_lambdas(self):
        return False
    
    def has_variables(self, bound_variables={}):
        return False
    
    def simplify(self):
        return self

    def graph_data(self, prefix=""):
        pass
    
    def graph(self):
        ast_graph = nx.DiGraph()
        nodes, node_labels, edges_list = self.graph_data()
        if len(node_labels)> 0:
            node_labels[0] = "Root: " + node_labels[0]
        node_data = [(node, dict(label=label)) for node, label in zip(nodes, node_labels)]
        ast_graph.add_nodes_from(node_data)
        ast_graph.add_edges_from(edges_list)
        return ast_graph

    def __eq__(a, b):
        if a.INDICATOR != b.INDICATOR:
            return wrap_with_token(False)
        return wrap_with_token(a.value == b.value)

    def __lt__(a, b):
        return wrap_with_token(a.value < b.value)

    def __gt__(a, b):
        return wrap_with_token(a.value > b.value)

    def __neg__(self):
        if isinstance(self, IntegerToken) or isinstance(self, UnaryToken):
            return wrap_with_token(-self.value)
        # previous also should process the double negation
        print("unexpected negation with", str(self))

    def __not__(self):
        if isinstance(self, BooleanToken) or isinstance(self, UnaryToken):
            return wrap_with_token(not self.value)
        print("unexpected not with", str(self))

    def __add__(a, b):
        return wrap_with_token(a.value + b.value)

    def __sub__(a, b):
        return wrap_with_token(a.value - b.value)

    def __mul__(a, b):
        return wrap_with_token(a.value * b.value)

    def __floordiv__(a, b):
        # this should work after I'll fix unary token value property
        value = np.sign(a.value)*np.sign(b.value)*(np.abs(a.value) // np.abs(b.value)) 
        value = int(value)
        return wrap_with_token(value)

    def __truediv__(a, b):
        print("b", b)
        return a // b

    def __mod__(a, b):
        value = np.sign(a.value)*np.sign(b.value)*(np.abs(a.value) % np.abs(b.value)) 
        value = int(value)
        return wrap_with_token(value)

# '=': lambda x, y: x == y,

# '|': lambda x, y: x or y if isinstance(x, bool) and isinstance(y, bool) else None,
# '&': lambda x, y: x and y if isinstance(x, bool) and isinstance(y, bool) else None,

# '.': lambda x, y: x + y if isinstance(x, str) and isinstance(y, str) else None,

# 'T': lambda x, y: y[:x] if isinstance(x, int) and isinstance(y, str) else None,
# 'D': lambda x, y: y[x:] if isinstance(x, int) and isinstance(y, str) else None,
        
        
class BooleanToken(CommonToken):
    def __init__(self, value):
        if value:
            self.INDICATOR = "T"
            self.value = True
        else:
            self.INDICATOR = "F"
            self.value = False
        self.body = ""

    @classmethod
    def is_match(cls, text):
        return (len(text) == 1) and (text[0] in "TF")

    @classmethod
    def parse(cls, token_str):
        indicator = token_str[0]
        assert indicator in "TF"
        value = indicator == "T"
        
        return BooleanToken(value)
    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}"

    def __call__(self):
        return self

    def show(self):
        return self.value

    def graph_data(self, prefix=""):
        return [prefix + self.INDICATOR], [self.INDICATOR], []

    def substitute(self, number, token):
        pass

    def __or__(a, b):
        return wrap_with_token(a.value | b.value)

    def __and__(a, b):
        return wrap_with_token(a.value & b.value)

    # def __repr__(self):
    #     class_name = self.__class__.__name__
    #     return f"{class_name}<{self.INDICATOR}, {self.NUM_PARAMETERS}>({self.value})"




STRING_DECODING_RULE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&\'()*+,-./:;<=>?@[\\]^_`|~ \n"
string_decoder_dict = {chr(i + 33): ch for i, ch in enumerate(STRING_DECODING_RULE)}
string_encoder_dict = {ch : chr(i + 33) for i, ch in enumerate(STRING_DECODING_RULE)}

"""
The Cult of the Bound variable seems to use a system similar to ASCII to encode characters, but ordered slightly differently. Specifically, ASCII codes 33 to 126 from the body can be translated to human readable text by converting them according to the following order:

"""

def decode(string):
    return "".join([string_decoder_dict[ch] for ch in string])

def encode(string):
    return "".join([string_encoder_dict[ch] for ch in string])


# .	String concatenation	B. S4% S34 -> "test"
# T	Take first x chars of string y	BT I$ S4%34 -> "tes"
# D	Drop first x chars of string y	BD I$ S4%34 -> "t"

class StringToken(CommonToken):
    INDICATOR = "S"
    def __init__(self, value, body=None):
        self.value = value
        if body is None:
            self.body = encode(value)
        else:
            self.body = body
        
    @classmethod
    def is_match(cls, text):
        return (len(text) >= 1) and (text[0] == "S")

    @classmethod
    def parse(cls, token_str):
        indicator = token_str[0]
        assert indicator == cls.INDICATOR
        body = token_str[1:]
        value = decode(body)
        
        return cls(value, body=body)

    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}({self.value})"


    def __call__(self):
        return self

    def show(self):
        return self.value
    
    def graph_data(self, prefix=""):
        return [ prefix+"s" ], [ repr(self.value) ], []
    def substitute(self, number, token):
        pass

    def take(self, int_token):
        n = int_token.value
        # print(n)
        return StringToken(self.value[:n])

    def drop(self, int_token):
        n = int_token.value
        return StringToken(self.value[n:])
    
    def concat(self, string_token):
        return StringToken(self.value + string_token.value)



def to_base94(number, base_number=94, zero_char='!'):
    result = []
    if number < 0:
        print("got negative number")
        number = - number
    
    while number:
        result += chr(ord(zero_char) + number % base_number)
        number //= base_number
    return "".join(result[::-1]) or zero_char


def from_base94(base94_str, base_number=94, zero_char='!'):
    number = 0
    for ch in base94_str:
        number *= base_number
        number += ord(ch) - ord(zero_char)
    return number

# '+': lambda x, y: x + y if isinstance(x, int) and isinstance(y, int) else None,
# '-': lambda x, y: x - y if isinstance(x, int) and isinstance(y, int) else None,
# '*': lambda x, y: x * y if isinstance(x, int) and isinstance(y, int) else None,
# '/': lambda x, y: np.sign(x)*np.sign(y)*(np.abs(x) // np.abs(y)) if isinstance(x, int) and isinstance(y, int) else None,

# '%': lambda x, y: np.sign(x)*np.sign(y)*(np.abs(x) % np.abs(y)) if isinstance(x, int) and isinstance(y, int) else None,
# '<': lambda x, y: x < y if isinstance(x, int) and isinstance(y, int) else None,
# '>': lambda x, y: x > y if isinstance(x, int) and isinstance(y, int) else None,

class IntegerToken(CommonToken):
    INDICATOR = "I"
    def __init__(self, value, body=None):
        self.value = value
        self.body = to_base94(value) if body is None else body

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)
    
    @classmethod
    def parse(cls, token_str):
        indicator = token_str[0]
        assert indicator == cls.INDICATOR
        body = token_str[1:]
        value = from_base94(body)
        
        return cls(value, body=body)

    def __call__(self):
        return self

    def show(self):
        return self.value

    def graph_data(self, prefix=""):
        return [ prefix+"i" ], [ str(self.value) ], []

    def substitute(self, number, token):
        pass



class UnaryToken(CommonToken):
    NUM_PARAMETERS = 1
    INDICATOR = "U"
    TOKEN_EXPRESSIONS = {
        '-': lambda x: -x,
        '!': lambda x: not x,
        '#': lambda x: from_base94(x), # from_base94(x), # integer token
        '$': lambda x: decode(x), # to_base94(x) #string token
    }

    def __init__(self, name, parameters):
        # assert self.is_match(start_token)
        self.name = name #start_token[1:]
        if isinstance(parameters, list):
            parameters = parameters[0]
        self.parameter = parameters
        self.cached_parameter = None

    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}{self.name}({repr(self.parameter)})"

    @property
    def value(self):
        # print("in value")
        # TODO: compute value for token it holds
        inner_token = self.parameter()
        # print(repr(inner_token))
        # inner_token = self()
        inner_value = self.TOKEN_EXPRESSIONS[self.name](inner_token.value)
        return inner_value

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)

    def __call__(self):
        # print("in call", str(self))
        if self.cached_parameter is not None:
            return self.cached_parameter
        computed_parameter = self.parameter()
        # print("param computed", str(computed_parameter))
        # jf self.name == "-" and 
        # print("param", self.name)
        # if self.name == "#":
        #     computed_parameter = wrap_with_token(computed_parameter)
        # elif self.name == "$":
        #     # print("param", computed_parameter)
        #     computed_parameter = wrap_with_token(computed_parameter)
        self.cached_parameter = self.TOKEN_EXPRESSIONS[self.name](computed_parameter)
        return self.cached_parameter

    def __str__(self):
        data = [f"{self.INDICATOR}{self.name}", str(self.parameter)]
        return " ".join(data)

    def show(self):
        a_str = self.parameter.show()
        return ["U" + self.name, a_str]

    def simplify(self):
        if self.parameter.has_lambdas():
            self.parameter = self.parameter.simplify()
        else:
            value = self.parameter()
            self.parameter = wrap_with_token(value)
        return self

    def has_lambdas(self):
        return self.parameter.has_lambdas()

    def has_variables(self, bound_variables={}):
        return self.parameter.has_variables(bound_variables)
    
    def graph_data(self, prefix=""):
        nodes, node_labels, edges_list = self.parameter.graph_data(prefix + "u")
        return [ prefix + "u" ] + nodes, [ f"{self.INDICATOR}{self.name}" ] + node_labels, [(prefix + "u", nodes[0])] + edges_list

    def substitute(self, number, token):
        # self.parameter.substitute(number, token)
        if self.parameter.INDICATOR == 'v' and self.parameter.number == number:
            # we have variable, let's replace it
            self.parameter = token
        else:
            self.parameter.substitute(number, token)
        # self.parameter.substitute(number, token)


class BinaryToken(CommonToken):
    
    NUM_PARAMETERS = 2
    INDICATOR = "B"
    """
    +	Integer addition	B+ I# I$ -> 5
    -	Integer subtraction	B- I$ I# -> 1
    *	Integer multiplication	B* I$ I# -> 6
    /	Integer division (truncated towards zero)	B/ U- I( I# -> -3
    
    %	Integer modulo	B% U- I( I# -> -1
    <	Integer comparison	B< I$ I# -> false
    >	Integer comparison	B> I$ I# -> true
    
    =	Equality comparison, works for int, bool and string	B= I$ I# -> false
    
    |	Boolean or	B| T F -> true
    &	Boolean and	B& T F -> false
    .	String concatenation	B. S4% S34 -> "test"
    T	Take first x chars of string y	BT I$ S4%34 -> "tes"
    D	Drop first x chars of string y	BD I$ S4%34 -> "t"
    $	Apply term x to y
    """
    TOKEN_EXPRESSIONS = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        # np.sign(x)*np.sign(y)*(np.abs(x) % np.abs(y)) 
        '%': lambda x, y: x % y,
        '<': lambda x, y: x < y,
        '>': lambda x, y: x > y,
        '=': lambda x, y: x == y,
        
        '|': lambda x, y: x | y,
        '&': lambda x, y: x & y,
        
        '.': lambda x, y: x.concat(y),
        
        'T': lambda x, y: y.take(x),
        'D': lambda x, y: y.drop(x),
        
        '$': lambda x, y: x.apply(y),
        # '$': lambda x, y: x.apply(y),
        # '$': lambda x, y: x.apply(y),
    }

    def __init__(self, name, parameters):
        # self.indicator = "B"
        # assert self.is_match(start_token)
        self.name = name #start_token[1:]
        self.parameters = parameters
        self.cached_value = None
        
    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)

    def call_bind(self):
        if self.cached_value is not None:
            return self.cached_value
        assert self.name == "$"
        # self.parameters[0].simplify()
        # todo:
        # print("after simplify is called", self.parameters)
        # print(self.parameters[0]())
        assert self.parameters[0].INDICATOR == 'L', str(self)
        lambda_abstraction = self.parameters[0]
        substituted_token = self.parameters[1]
        substituted_token = substituted_token.simplify()
        token = lambda_abstraction.apply(substituted_token)
        # print("after apply", repr(token))
        if token.has_lambdas() or token.has_variables():
            token = token.simplify()
        else:
            token = token()
        self.cached_value = token
        return self.cached_value
    
    def __call__(self):
        if self.cached_value is not None:
            return self.cached_value
        #if self.name == "$":
        #    return self.simplify()
        #    #print(self.parameters)
        #    #return self.call_bind()
        self.parameters[0] = self.parameters[0].simplify()
        if self.name == "$" and self.parameters[0].INDICATOR == 'L':
            return self.call_bind()
            # computed_parameter0 = self.parameters[0]
        else:
            computed_parameter0 = self.parameters[0]()
            # self.parameters[0] = computed_parameter0
            
        # add optimization for boolean values
        if self.name == "|" and computed_parameter0.value:
            self.cached_value = wrap_with_token(True)
            return self.cached_value
        if self.name == "&" and not computed_parameter0.value:
            self.cached_value = wrap_with_token(False)
            return self.cached_value
        # do regular stuff
        self.parameters[1] = self.parameters[1].simplify()
        computed_parameter1 = self.parameters[1]()
        
        self.cached_value = self.TOKEN_EXPRESSIONS[self.name](computed_parameter0, computed_parameter1)
        return self.cached_value

    def __str__(self):
        data = [f"{self.INDICATOR}{self.name}"]+[str(p) for p in self.parameters]
        return " ".join(data)

    def __repr__(self):
        # class_name = self.__class__.__name__
        # parameter_representations = [p for p in self.parameters]
        return f"{self.INDICATOR}{self.name}({self.parameters})"

    def show(self):
        a_str = self.parameters[0].show()
        b_str = self.parameters[1].show()
        return ["B"+self.name, a_str, b_str]

    def has_lambdas(self):
        return self.parameters[0].has_lambdas() or self.parameters[1].has_lambdas()
    
    def has_variables(self, bound_variables={}):
        print(self.parameters[0])
        return self.parameters[0].has_variables(bound_variables) or self.parameters[1].has_variables(bound_variables)

    def simplify(self):
        # print("run simplify", self.name, repr(self.parameters))
        # evaluate parameters if they are not 
        p0 = self.parameters[0]
        if p0.has_lambdas() or p0.has_variables():
            p0 = p0.simplify()
        else:
            p0 = p0()
            self.parameters[0] = p0
        p1 = self.parameters[1]
        if p1.has_lambdas() or p1.has_variables():
            p1 = p1.simplify()
        else:
            p1 = p1()
            self.parameters[1] = p1
        if self.name == "$" and p0.INDICATOR == "L":
            return self.call_bind()
        self.parameters = [p0, p1]
        # print("run simplify - end", self.name, self.parameters, self.cached_value)
        return self
        # print(repr(p0), repr(p1))
        #if self.name == "$" and p0.INDICATOR == "L":
        #    self.call_bind()

    def graph_data(self, prefix=""):
        nodes0, node_labels0, edges_list0 = self.parameters[0].graph_data(prefix + "b0")
        nodes1, node_labels1, edges_list1 = self.parameters[1].graph_data(prefix + "b1")
        nodes = [prefix + "b"] + nodes0 + nodes1
        node_labels = [f"{self.INDICATOR}{self.name}"] + node_labels0 + node_labels1
        edges_list = [(prefix + "b", nodes0[0]), (prefix + "b", nodes1[0])] + edges_list0 + edges_list1
        return nodes, node_labels, edges_list
    
    def substitute(self, number, token):
        if self.parameters[0].INDICATOR == 'v' and self.parameters[0].number == number:
            # we have variable, let's replace it
            self.parameters[0] = token
        else:
            self.parameters[0].substitute(number, token)
        if self.parameters[1].INDICATOR == 'v' and self.parameters[1].number == number:
            # we have variable, let's replace it
            self.parameters[1] = token
        else:
            self.parameters[1].substitute(number, token)
        #self.parameters[0].substitute(number, token)
        #self.parameters[1].substitute(number, token)



class IfToken(CommonToken):
    NUM_PARAMETERS = 3
    INDICATOR = "?"

    def __init__(self, name, parameters):
        # self.indicator = "B"
        # assert self.is_match(start_token)
        assert len(parameters) == self.NUM_PARAMETERS
        self.condition = parameters[0]
        self.t_value = parameters[1]
        self.f_value = parameters[2]

        self.cached_value = None
        self.substitutions_list = []

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 1) and (text[0] == cls.INDICATOR)
    
    def __call__(self):
        if self.cached_value is not None:
            return self.cached_value
        computed_parameter_condition = self.condition()
        if computed_parameter_condition:
            computed_parameter = self.t_value()
            self.cached_value = computed_parameter
        else:
            computed_parameter = self.f_value()
            self.cached_value = computed_parameter

        return self.cached_value

    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}[{repr(self.condition)}, {repr(self.t_value)}, {repr(self.f_value)}]"

    def __str__(self):
        data = [self.INDICATOR, str(self.condition), str(self.t_value), str(self.f_value)]
        return " ".join(data)

    def show(self):
        return {
            "if": self.condition.show(),
            "value_true": self.t_value.show(),
            "value_false": self.f_value.show(),
        }

    def has_lambdas(self):
        return self.condition.has_lambdas() or self.t_value.has_lambdas() or self.f_value.has_lambdas()

    def has_variables(self, bound_variables={}):
        return self.condition.has_variables(bound_variables) or self.t_value.has_variables(bound_variables) or self.f_value.has_variables(bound_variables)
    
    def simplify(self):
        if self.condition.has_lambdas() or self.condition.has_variables():
            self.condition = self.condition.simplify()
            print("if condition, simplify", self.condition)
        else:
            value = self.condition()
            self.condition = wrap_with_token(value)
            print("if condition, simpl2", value)
            if value:
                if self.t_value.has_lambdas() or self.t_value.has_variables():
                    self.t_value = self.t_value.simplify()
                else:
                    value = self.t_value()
                    self.t_value = wrap_with_token(value)
                self.f_value = wrap_with_token('any value')
            else:
                self.t_value = wrap_with_token('any value')
                if self.f_value.has_lambdas()  or self.f_value.has_variables():
                    self.f_value = self.f_value.simplify()
                else:
                    value = self.f_value()
                    self.f_value = wrap_with_token(value)
            return self
        # 
        if self.t_value.has_lambdas() or self.t_value.has_variables():
            self.t_value = self.t_value.simplify()
        else:
            value = self.t_value()
            self.t_value = wrap_with_token(value)
        if self.f_value.has_lambdas() or self.f_value.has_variables():
            self.f_value = self.f_value.simplify()
        else:
            value = self.f_value()
            self.f_value = wrap_with_token(value)
        return self

    def graph_data(self, prefix=""):
        nodes_c, node_labels_c, edges_list_c = self.condition.graph_data(prefix + "if.c")
        nodes_t, node_labels_t, edges_list_t = self.t_value.graph_data(prefix + "if.t")
        nodes_f, node_labels_f, edges_list_f = self.f_value.graph_data(prefix + "if.f")
        
        nodes = [prefix + "if"] + nodes_c + nodes_t + nodes_f
        node_labels = ["if condition"] + node_labels_c + node_labels_t + node_labels_f
        edges_list = [(prefix + "if", nodes_c[0]), (prefix + "if", nodes_t[0]), (prefix + "if", nodes_f[0])] + edges_list_c + edges_list_t + edges_list_f
        return nodes, node_labels, edges_list

    def substitute(self, number, token):
        # self.condition.substitute(number, token)
        # self.t_value.substitute(number, token)
        # self.f_value.substitute(number, token)
        if self.condition.INDICATOR == 'v' and self.condition.number == number:
            # we have variable, let's replace it
            self.condition = token
        else:
            self.condition.substitute(number, token)
        if self.t_value.INDICATOR == 'v' and self.t_value.number == number:
            # we have variable, let's replace it
            self.t_value = token
        else:
            self.t_value.substitute(number, token)
        if self.f_value.INDICATOR == 'v' and self.f_value.number == number:
            # we have variable, let's replace it
            self.f_value = token
        else:
            self.f_value.substitute(number, token)
    

class LambdaToken(CommonToken):
    INDICATOR = "L"
    NUM_PARAMETERS = 1
    def __init__(self, number, parameters):
        # self.number = from_base94(name)
        if isinstance(number, str):
            number = from_base94(number)
        self.number = number
        # self.body = body
        if isinstance(parameters, CommonToken):
            self.expression = parameters
        elif isinstance(parameters, list):
            self.expression = parameters[0]
        else:
            self.expression = wrap_with_token(parameters)

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)
    
    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}{self.number}->{repr(self.expression)}"
        
    @classmethod
    def parse(cls, start_token, parameters):
        # expression = parameters
        # indicator = text[0]
        number = from_base94(start_token[1:])
        # value = from_base94(body)
        
        return cls(number, parameters)

    def __str__(self):
        # print(self.number, type(self.number))
        encoded_number = to_base94(self.number)
        start_token = f"{self.INDICATOR}{encoded_number}"
        next_tokens = str(self.expression)
        return f"{start_token} {next_tokens}"

    def show(self):
        return {f"{self.INDICATOR}{self.number}": self.expression.show()}

    def has_lambdas(self):
        return True

    def has_variables(self, bound_variables={}):
        # todo: check if there are unbounded variables
        bound_vars_updated = {*bound_variables, self.number}
        return self.expression.has_variables(bound_vars_updated)
        # return True

    def graph_data(self, prefix=""):
        name = f"{self.INDICATOR}{self.number}"
        expr_nodes, expr_node_labels, expr_edges_list = self.expression.graph_data(prefix + name)
        
        nodes = [prefix + name] + expr_nodes
        node_labels = [f"lambda {self.number}"] + expr_node_labels
        edges_list = [(prefix + name, expr_nodes[0])] + expr_edges_list
        return nodes, node_labels, edges_list
    
    def simplify(self):
        if self.expression.has_lambdas() or self.expression.has_variables():
            self.expression = self.expression.simplify()
        else:
            value = self.expression()
            self.expression = wrap_with_token(value)
        return self

    def apply(self, token):
        if not isinstance(token, CommonToken):
            token = wrap_with_token(token)
        token = token.simplify()
        if self.expression.INDICATOR == "v":
            # print("comparing expression inficator with current one")
            # print(repr(self.expression), self.number)
            if self.expression.number == self.number:
                self.expression = token
            else:
                # print("call substitute with number", self.number, repr(self.expression), repr(token))
                self.expression.substitute(self.number, token)
        else:
            # print("call substitute in apply", self.number, repr(self.expression), repr(token))
            self.expression.substitute(self.number, token)
        # if self.expression.INDICATOR == "v":
            
        # self.expression.substitute(self.number, token)
        return self.expression

    def substitute(self, number, token):
        if self.number != number:
            #self.expression.substitute(number, token)
            if self.expression.INDICATOR == "v" and self.expression.number == number:
                self.expression = token
            else:
                self.expression.substitute(number, token)
            # self.expression.substitute(number, token)
        # if this lambda's number != number, try to substitute in the body of the method
        

        # if self.condition.INDICATOR == 'v' and self.condition.number == number:
        #     # we have variable, let's replace it
        #     self.condition = token
        # if self.t_value.INDICATOR == 'v' and self.t_value.number == number:
        #     # we have variable, let's replace it
        #     self.t_value = token
        # if self.f_value.INDICATOR == 'v' and self.f_value.number == number:
        #     # we have variable, let's replace it
        #     self.f_value = token
    #def apply(self, token):
    #    self.expression()
    #    pass
    # def __call__(self, variables={}):
    #     ???
    #     if self.cached_value is not None:
    #         return self.cached_value
    #     computed_parameter_condition = self.condition()
    #     if computed_parameter_condition:
    #         computed_parameter = self.t_value()
    #         self.cached_value = computed_parameter
    #     else:
    #         computed_parameter = self.f_value()
    #         self.cached_value = computed_parameter

    #     return self.cached_value
    

class VariableToken(CommonToken):
    INDICATOR = "v"
    def __init__(self, number):
        self.number = number
        self.body = to_base94(number)
        self.substitution_list = []

    @classmethod
    def is_match(cls, token_str):
        return (len(token_str) >= 2) and (token_str[0] == cls.INDICATOR)

    @classmethod
    def parse(cls, token_str):
        # indicator = text[0]
        assert cls.is_match(token_str)
        body = token_str[1:]
        number = from_base94(body)
        return cls(number)

    def __str__(self):
        return F"{self.INDICATOR}{self.body}"
    
    def show(self):
        return F"{self.INDICATOR}{self.number}"

    def graph_data(self, prefix=""):
        name = f"{self.INDICATOR}{self.number}"

        nodes = [prefix + name]
        node_labels = [name] 
        edges_list = []
        return nodes, node_labels, edges_list

    def has_lambdas(self):
        return False

    def has_variables(self, bound_variables={}):
        return not self.number in bound_variables

    def simplify(self):
        # this is intentionally left blank
        return self

    def substitute(self, number, token):
        # this is intentionally left blank
        pass
        #self.substitution_list.append((number, token))

    # @property
    # def value(self):
    #     return self.number
    def __repr__(self):
        # class_name = self.__class__.__name__
        return f"{self.INDICATOR}{self.number}"

    def __call__(self):
        print(111)
        return self

