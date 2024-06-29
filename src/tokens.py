from dataclasses import dataclass
import numpy as np

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
    @property
    def is_complex(self):
        return self.NUM_PARAMETERS > 0

    @property
    def has_lambdas(self):
        return False
    
    def simplify(self):
        pass


class BooleanToken(CommonToken):
    def __init__(self, value):
        self.value = value
        self.indicator = "T" if value else "F"
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

    def __call__(self, variables={}):
        return self.value

    def show(self):
        return self.value
        


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

    def __call__(self, variables={}):
        return self.value

    def show(self):
        return self.value



def to_base94(number, base_number=94, zero_char='!'):
    result = []
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

    def __call__(self, variables={}):
        return self.value

    def show(self):
        return self.value


def wrap_with_token(value):
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
    return token


class UnaryToken(CommonToken):
    NUM_PARAMETERS = 1
    INDICATOR = "U"
    TOKEN_EXPRESSIONS = {
        '-': lambda x: -x,
        '!': lambda x: not x,
        '#': lambda x: from_base94(x), # from_base94(x),
        '$': lambda x: decode(x), # to_base94(x)
    }

    def __init__(self, name, parameters):
        # assert self.is_match(start_token)
        self.name = name #start_token[1:]
        if isinstance(parameters, list):
            parameters = parameters[0]
        self.parameter = parameters
        self.cached_value = None

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)

    def __call__(self, variables={}):
        computed_parameter = self.parameter(variables=variables)
        if self.cached_value is not None:
            return self.cached_value
        print("param", self.name)
        if self.name == "#":
            computed_parameter = wrap_with_token(computed_parameter).body
        elif self.name == "$":
            # print("param", computed_parameter)
            computed_parameter = wrap_with_token(computed_parameter).body
        self.cached_value = self.TOKEN_EXPRESSIONS[self.name](computed_parameter)
        return self.cached_value
    def __str__(self):
        data = [f"{self.INDICATOR}{self.name}", str(self.parameter)]
        return " ".join(data)

    def show(self):
        a_str = self.parameter.show()
        return ["U" + self.name, a_str]

    def simplify(self):
        if self.parameter.has_lambdas:
            self.parameter.simplify()
        else:
            value = self.parameter()
            self.parameter = wrap_with_token(value)
    @property
    def has_lambdas(self):
        return self.parameter.has_lambdas




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
        '+': lambda x, y: x + y if isinstance(x, int) and isinstance(y, int) else None,
        '-': lambda x, y: x - y if isinstance(x, int) and isinstance(y, int) else None,
        '*': lambda x, y: x * y if isinstance(x, int) and isinstance(y, int) else None,
        '/': lambda x, y: np.sign(x)*np.sign(y)*(np.abs(x) // np.abs(y)) if isinstance(x, int) and isinstance(y, int) else None,
        
        '%': lambda x, y: np.sign(x)*np.sign(y)*(np.abs(x) % np.abs(y)) if isinstance(x, int) and isinstance(y, int) else None,
        '<': lambda x, y: x < y if isinstance(x, int) and isinstance(y, int) else None,
        '>': lambda x, y: x > y if isinstance(x, int) and isinstance(y, int) else None,
        '=': lambda x, y: x == y,
        
        '|': lambda x, y: x or y if isinstance(x, bool) and isinstance(y, bool) else None,
        '&': lambda x, y: x and y if isinstance(x, bool) and isinstance(y, bool) else None,
        
        '.': lambda x, y: x + y if isinstance(x, str) and isinstance(y, str) else None,
        
        'T': lambda x, y: y[:x] if isinstance(x, int) and isinstance(y, str) else None,
        'D': lambda x, y: y[x:] if isinstance(x, int) and isinstance(y, str) else None,
        
        '$': lambda x, y: x.apply(y),
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
    
    def __call__(self, variables={}):
        computed_parameter0 = self.parameters[0](variables=variables)
        # add optimization for boolean values
        if self.name == "|" and computed_parameter0:
            self.cached_value = True
            return self.cached_value
        if self.name == "&" and not computed_parameter0:
            self.cached_value = False
            return self.cached_value
        # do regular stuff
        computed_parameter1 = self.parameters[1](variables=variables)
        if self.cached_value is not None:
            return self.cached_value
        self.cached_value = self.TOKEN_EXPRESSIONS[self.name](computed_parameter0, computed_parameter1)
        return self.cached_value
    def __str__(self):
        data = [f"{self.INDICATOR}{self.name}"]+[str(p) for p in self.parameters]
        return " ".join(data)
    
    def show(self):
        a_str = self.parameters[0].show()
        b_str = self.parameters[1].show()
        return ["B"+self.name, a_str, b_str]
    @property
    def has_lambdas(self):
        return self.parameters[0].has_lambdas or self.parameters[1].has_lambdas

    def simplify(self):
        # evaluate parameters if they are not 
        p0 = self.parameters[0]
        if p0.has_lambdas:
            p0.simplify()
        else:
            p0_value = p0()
            p0 = wrap_with_token(p0_value)
            self.parameters[0] = p0
        p1 = self.parameters[1]
        if p1.has_lambdas:
            p1.simplify()
        else:
            p1_value = p1()
            p1 = wrap_with_token(p1_value)
            self.parameters[1] = p1
        


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

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 1) and (text[0] == cls.INDICATOR)
    
    def __call__(self, variables={}):
        if self.cached_value is not None:
            return self.cached_value
        computed_parameter_condition = self.condition(variables=variables)
        if computed_parameter_condition:
            computed_parameter = self.t_value(variables=variables)
            self.cached_value = computed_parameter
        else:
            computed_parameter = self.f_value(variables=variables)
            self.cached_value = computed_parameter

        return self.cached_value

    def __str__(self):
        data = [self.INDICATOR, str(self.condition), str(self.t_value), str(self.f_value)]
        return " ".join(data)

    def show(self):
        return {
            "if": self.condition.show(),
            "value_true": self.t_value.show(),
            "value_false": self.f_value.show(),
        }

    @property
    def has_lambdas(self):
        return self.condition.has_lambdas or self.t_value.has_lambdas or self.f_value.has_lambdas
    def simplify(self):
        if self.condition.has_lambdas:
            self.condition.simplify()
        else:
            value = self.condition()
            self.condition = wrap_with_token(value)
            if value:
                if self.t_value.has_lambdas:
                    self.t_value.simplify()
                else:
                    value = self.t_value()
                    self.t_value = wrap_with_token(value)
                self.f_value(wrap_with_token('any value'))
            else:
                self.t_value(wrap_with_token('any value'))
                if self.f_value.has_lambdas:
                    self.f_value.simplify()
                else:
                    value = self.f_value()
                    self.f_value = wrap_with_token(value)
            return
        # 
        if self.t_value.has_lambdas:
            self.t_value.simplify()
        else:
            value = self.t_value()
            self.t_value = wrap_with_token(value)
        if self.f_value.has_lambdas:
            self.f_value.simplify()
        else:
            value = self.f_value()
            self.f_value = wrap_with_token(value)
    

class LambdaToken(CommonToken):
    INDICATOR = "L"
    NUM_PARAMETERS = 1
    def __init__(self, number, parameters):
        # self.number = from_base94(name)
        if isinstance(number, str):
            number = from_base94(number)
        self.number = number
        # self.body = body
        self.expression = parameters[0]

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)
    
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
    @property
    def has_lambdas(self):
        return True
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
    

