from typing import List
import tokens

def partial_parse(token_list: List[str]):
    """returns next parsed token and new start_pos
    """
    if len(token_list) < 1:
        return None
    next_token_str = token_list[0]
    # print(next_token_str, token_list[1:])
    for simple_token_class in [tokens.BooleanToken, tokens.StringToken, tokens.IntegerToken, tokens.VariableToken]:
        if simple_token_class.is_match(next_token_str):
            next_token = simple_token_class.parse(next_token_str)
            return next_token, token_list[1:]
    for op_token_class in [tokens.UnaryToken, tokens.BinaryToken, tokens.IfToken, tokens.LambdaToken]:
        if op_token_class.is_match(next_token_str):
            remaining_tokens = token_list[1:]
            parameters = []
            for k in range(op_token_class.NUM_PARAMETERS):
                op_parameter, remaining_tokens = partial_parse(remaining_tokens)
                parameters.append(op_parameter)
            op_token = op_token_class(next_token_str[1:], parameters)
            # print("parsed token", op_token, op_token_class.__name__)
            return op_token, remaining_tokens
            # op_parameter1, remaining_tokens = partial_parse(token_list[1:])
    
    # print(token_list)
    raise Exception("Not implemented yet")


def parse_text(text):
    tokens_list = text.split()
    expression, unprocessed_tokens = partial_parse(tokens_list)
    assert len(unprocessed_tokens) == 0
    return expression