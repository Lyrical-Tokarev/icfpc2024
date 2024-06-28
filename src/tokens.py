from dataclasses import dataclass


@dataclass
class CommonToken:
    """Class for tokens"""
    INDICATOR = None
    def __init__(self, indicator: str, body: str, value=None):
        self.indicator = indicator
        self.body = body
        self.value = value
    
    @classmethod
    def is_match(cls, text):
        # TODO: add condition "(!') up to and including code 126 ('~)"
        return len(text) >= 1 

    @classmethod
    def parse(cls, text):
        # indicator = text[0]
        assert cls.is_match(text)
        body = text[1]
        return CommonToken(body)

    def __str__(self):
        return f"{self.INDICATOR}{self.body}"
    


class BooleanToken(CommonToken):
    @staticmethod
    def is_match(text):
        return (len(text) == 2) and (text[0] in "TF")

    @classmethod
    def parse(cls, text):
        indicator = text[0]
        body = text[1:]
        value = None
        if indicator == "T":
            value = True
        if indicator == "F":
            value = False
        return BooleanToken(indicator, body, value=value)

    def __call__(self):
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
    @staticmethod
    def is_match(text):
        return (len(text) >= 1) and (text[0] == "S")

    @classmethod
    def parse_text(cls, text):
        return cls(indicator="S", body=encode(text), value=text)

    @classmethod
    def parse(cls, text):
        indicator = text[0]
        body = text[1:]
        value = decode(body)
        
        return cls(indicator, body, value=value)

    def __call__(self):
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
    def __init__(self, number):
        self.indicator = "I"
        self.body = to_base94(number)
        self.value = number

    @staticmethod
    def is_match(text):
        return (len(text) >= 2) and (text[0] == "I")
    
    @classmethod
    def parse(cls, text):
        indicator = text[0]
        body = text[1:]
        value = from_base94(body)
        
        return cls(indicator, body, value=value)

    def __call__(self):
        return self.value


class UnaryToken(CommonToken):
    NUM_PARAMETERS = 1
    INDICATOR = "B"
    TOKEN_EXPRESSIONS = {
        '-': lambda x: -x,
        '!': lambda x: not x,
        '#': lambda x: from_base94(x),
        '$': lambda x: to_base94(x)
    }

    def __init__(self, name, parameters):
        # assert self.is_match(start_token)
        self.name = name #start_token[1:]
        self.parameter = parameters[0]
        self.cached_value = None

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)

    def __call__(self):
        computed_parameter = self.parameter()
        if self.cached_value is not None:
            return self.cached_value
        self.cached_value = self.TOKEN_EXPRESSIONS[self.name](computed_parameter)
        return self.cached_value
    def __str__(self):
        data = [f"{self.INDICATOR}{self.name}", str(self.parameter)]
        return " ".join(data)


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
        '/': lambda x, y: x // y if isinstance(x, int) and isinstance(y, int) else None,
        
        '%': lambda x, y: x % y if isinstance(x, int) and isinstance(y, int) else None,
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
    
    def __call__(self):
        computed_parameter0 = self.parameters[0]()
        computed_parameter1 = self.parameters[1]()
        if self.cached_value is not None:
            return self.cached_value
        self.cached_value = self.TOKEN_EXPRESSIONS[self.name](computed_parameter0, computed_parameter1)
        return self.cached_value
    def __str__(self):
        data = [f"{self.INDICATOR}{self.name}"]+[str(p) for p in self.parameters]
        return " ".join(data)


class IfToken(CommonToken):
    NUM_PARAMETERS = 3
    INDICATOR = "?"

    def __init__(self, parameters):
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

    def __str__(self):
        data = [self.INDICATOR, str(self.condition), str(self.t_value), str(self.f_value)]
        return " ".join(data)
        
    

class LambdaToken(CommonToken):
    INDICATOR = "L"
    def __init__(self, number, body):
        self.number = number
        self.body = body

    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)
    @classmethod
    def parse(cls, start_token, body):
        # indicator = text[0]
        number = from_base94(start_token[1:])
        # value = from_base94(body)
        
        return cls(number, body)

    def __str__(self):
        encoded_number = to_base94(self.number)
        start_token = f"{self.INDICATOR}{encoded_number}"
        next_token = str(self.body)
        return f"{start_token} {next_token}"
    

class VariableToken(CommonToken):
    INDICATOR = "v"
    def __init__(self, number):
        self.number = number
        self.body = to_base94(number)
    
    @classmethod
    def is_match(cls, text):
        return (len(text) >= 2) and (text[0] == cls.INDICATOR)
    

