from transformers import AutoTokenizer
from typing import List, Tuple
import random
import numpy as np 
import string
import warnings
import torch

# from random_word import RandomWords
# r = RandomWords()

cap_alpha_lst = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
digit_lst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
warned = False

or_dict = torch.load("util/or_dict.pt")
de_dict = torch.load("util/de_dict.pt")
en_dict = torch.load("util/en_dict.pt")
es_dict = torch.load("util/es_dict.pt")
fr_dict = torch.load("util/fr_dict.pt")
ko_dict = torch.load("util/ko_dict.pt")
zh_dict = torch.load("util/zh_dict.pt")

country_capital_dict = torch.load("util/country_capital_dict.pt")
country_continent_dict = torch.load("util/country_continent_dict.pt")
country_currency_dict = torch.load("util/country_currency_dict.pt")
country_eng_ch_dict = torch.load("util/country_eng_ch_dict.pt")

digit_char_map = {
    0: 'a',
    1: 'b',
    2: 'c',
    3: 'd',
    4: 'e',
    5: 'f',
    6: 'g',
    7: 'h',
    8: 'i',
    9: 'j',
    10: 'k',
    11: 'l',
    12: 'm',
    13: 'n',
    14: 'o',
    15: 'p',
    16: 'q',
    17: 'r',
    18: 's',
    19: 't'
}

def get_randint():
    while True:
        num = random.randint(3, 9)
        if num != 5:
            return num

def next_digit(digit: int):
    return str((digit + 1) % 10)

def digit_translation(digit: int):
    translation_dict = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four", 
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine"
    }
    return translation_dict[digit]

def next_translation(digit: int):
    return digit_translation(int(next_digit(digit)))

def left_pad(number: int, length: int) -> str:
    return str(number).zfill(length)

# All functiosn that generate task string should have examples as the first variable (can be unused)
def generate_task_copy(examples: List[str], tokenizer: AutoTokenizer, length: int, copy_idx: int, cap=False, digit=False, op='digits', low=0, high=9, symbol='@', symbol2='=') -> str:
    sequence = []
    while len(sequence) < length:
        if op == 'digits':
            operand = random.randint(low, high)
        elif op == 'letter':
            operand = random.choice(string.ascii_letters)
        elif op == 'ot_letters':
            bool1 = True
            while bool1:
                l1 = random.choice('abcdefghijklmnopqrstuvwxyz')
                l2 = random.choice('abcdefghijklmnopqrstuvwxyz')
                operand = l1+l2
                if len(tokenizer(operand, add_special_tokens=False).input_ids) == 1:
                    bool1 = False
        if operand not in sequence: 
            sequence.append(operand)
    if cap:
        sequence_str = "".join([f"{alpha}{operand}" for alpha, operand in zip(cap_alpha_lst[:length], sequence)])
    elif digit:
        sequence_str = "".join([f"{digit}{operand}" for digit, operand in zip(digit_lst[:length], sequence)])
    else:
        sequence_str = f"{symbol}".join(map(str, sequence))
    
    answer = sequence[copy_idx]
    out_str = f"{sequence_str}{symbol2}{answer}"
    symbol2_idx = out_str.index(symbol2)
    global warned
    if len(tokenizer(out_str[symbol2_idx:symbol2_idx+len(symbol2)], add_special_tokens=False).input_ids) > 1:
        if not warned:
            warnings.warn("symbol2 is not a single token")
    return out_str

def generate_task_AplusC(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, C: int, symbol='@', symbol2='=') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    return f"{A}{symbol}{B}{symbol2}{A+C}"

def generate_task_BplusC(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, C: int, symbol='@', symbol2='=') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    return f"{A}{symbol}{B}{symbol2}{B+C}"

def generate_task_copyA(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, AplusB_num_digits=None, symbol='@', symbol2='=')->str:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if not AplusB_num_digits is None:
            if len(str(A+B)) != AplusB_num_digits:
                continue
        return f"{A}{symbol}{B}{symbol2}{A}"

def generate_task_copyB(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, AplusB_num_digits=None, symbol='@', symbol2='=')->str:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if not AplusB_num_digits is None:
            if len(str(A+B)) != AplusB_num_digits:
                continue
        return f"{A}{symbol}{B}{symbol2}{B}"

def generate_task_AplusB(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, AplusB_num_digits=None, symbol='@', symbol2='=')->str:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if not AplusB_num_digits is None:
            if len(str(A+B)) != AplusB_num_digits:
                continue
        return f"{A}{symbol}{B}{symbol2}{A+B}"

def generate_task_AplusB_t(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, AplusB_num_digits=None, language="en", symbol='@', symbol2='=')->str:
    num_dict = globals()[f"{language}_dict"]
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if not AplusB_num_digits is None:
            if len(str(A+B)) != AplusB_num_digits:
                continue
        out_num = A+B
        out = num_dict[out_num]
        # if language == "en":
        #     out = out.replace("-", " ")
        return f"{A}{symbol}{B}{symbol2}{out}"

def generate_task_AplusBplusC(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, C: int, symbol='@', symbol2='=') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    return f"{A}{symbol}{B}{symbol2}{A+B+C}"

def generate_task_AplusB_plusC_mod(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, C:int, mod=10, alpha=False, symbol='@', symbol2='=') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    if alpha:
        return f"{A}{symbol}{B}{symbol2}{digit_char_map[(A+B)%mod]}"
    return f"{A}{symbol}{B}{symbol2}{(A+B+C)%mod}"

def generate_task_ndigit(examples: List[str], tokenizer: AutoTokenizer, bool=False) -> str:
    while True:
        digit = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ndigit = next_digit(digit)
        if bool:
            if f"{digit}->" not in "\n".join(examples):
                return f"{digit}->{ndigit}"
        else:
            return f"{digit}->{ndigit}"


def generate_task_digitt(examples: List[str], tokenizer: AutoTokenizer, bool=False) -> str:
    while True:
        digit = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        tdigit = digit_translation(digit)
        if bool:
            if f"{digit}->" not in "\n".join(examples):
                return f"{digit}->{tdigit}"
        else:
            return f"{digit}->{tdigit}"

def generate_task_ndigitt(examples: List[str], tokenizer: AutoTokenizer, bool=False) -> str:
    while True:
        digit = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ndigitt = next_translation(digit)
        if bool:
            if f"{digit}->" not in "\n".join(examples):
                return f"{digit}->{ndigitt}"
        else:
            return f"{digit}->{ndigitt}"
        
def generate_task_ABC_copyA(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, length: int, symbol='@') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    C = random.randint(low, high)

    return f"{A}{symbol}{B}{symbol}{C}->{left_pad(A, length)}"

def generate_task_ABC_AplusB(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, length: int, symbol='@') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    C = random.randint(low, high)

    return f"{A}{symbol}{B}{symbol}{C}->{left_pad(A+B, length)}"

def generate_task_ABC_AplusBplusC(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, length: int, symbol='@') -> str:
    A = random.randint(low, high)
    B = random.randint(low, high)
    C = random.randint(low, high)

    return f"{A}{symbol}{B}{symbol}{C}->{left_pad(A+B+C, length)}"

def generate_task_country_capital(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            capital = country_capital_dict[country]
            return f"{country}{symbol2}{capital}"

def generate_task_country_continent(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            continent = country_continent_dict[country]
            return f"{country}{symbol2}{continent}"

def generate_task_country_letter(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            num_letters = len(country.replace(" ", ""))
            return f"{country}{symbol2}{num_letters} letters"
        
def generate_task_country_upper(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            upper = country.upper()
            return f"{country}{symbol2}{upper}"

def generate_task_country_zh(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            name_zh = country_eng_ch_dict[country]
            return f"{country}{symbol2}{name_zh}"

def generate_task_country_currency(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            currency = country_currency_dict[country]
            return f"{country}{symbol2}{currency}"

def generate_task_first_letter(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        word = r.get_random_word()
        if len(word) > 1 and word[0].lower() != word[-1].lower():
            return f"{word}{symbol2}{word[0].lower()}"

def generate_task_last_letter(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        word = r.get_random_word()
        if len(word) > 1 and word[0].lower() != word[-1].lower():
            return f"{word}{symbol2}{word[-1].lower()}"

def generate_task_first_letter_cap(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        word = r.get_random_word()
        if len(word) > 1 and word[0].lower() != word[-1].lower():
            return f"{word}{symbol2}{word[0].upper()}"

def generate_task_num_self(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    num = get_randint()
    return f"{num}{symbol2}{num}"

def generate_task_num_negate(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    num = get_randint()
    return f"{num}{symbol2}{-1*num}"

def generate_task_num_p1(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    num = get_randint()
    return f"{num}{symbol2}{num+1}"

def generate_task_num_m1(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    num = get_randint()
    return f"{num}{symbol2}{num-1}"

def generate_task_num_t2(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    num = get_randint()
    return f"{num}{symbol2}{num*2}"

def generate_task_num_sq(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    num = get_randint()
    return f"{num}{symbol2}{num**2}"

def generate_task_last_letter_cap(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> str:
    while True:
        word = r.get_random_word()
        if len(word) > 1 and word[0].lower() != word[-1].lower():
            return f"{word}{symbol2}{word[-1].upper()}"

def generate_task_linear(examples: List[str], tokenizer: AutoTokenizer, range_min: int, range_max: int, a: int, b: int, symbol2=',') -> str:
    x = random.randint(range_min, range_max)
    y = int(a * x + b)
    return f"{x}{symbol2}{y}"
        
def question_digit(examples: List[str], input: "random") -> Tuple:
    if input == "random":
        while True:
            digit = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            if f"{digit}->" not in "\n".join(examples):
                next_digit = (digit + 1) % 10
                return f"{digit}->", [str(next_digit), digit_translation(digit), digit_translation(next_digit), str(digit)]

# All functions that generate question string should have examples as the first variable
def question_AB1(examples: List[str], tokenizer: AutoTokenizer, low : int, high : int, AplusB_num_digits=None, symbol='@', symbol2='=') -> Tuple:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if not AplusB_num_digits is None:
            if len(str(A+B)) != AplusB_num_digits:
                continue
        if f"{A}{symbol}{B}" not in "\n".join(examples):
            return f"{A}{symbol}{B}{symbol2}", [str(A), str(B), str(A+B)]
        
def question_1(examples: List[str], tokenizer: AutoTokenizer, low : int, high : int, C : int, symbol='@', symbol2='=') -> Tuple:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if f"{A}{symbol}{B}" not in "\n".join(examples):
            return f"{A}{symbol}{B}{symbol2}", [str(A), str(B), str(A+C), str(B+C)]

def question_2(examples: List[str], tokenizer: AutoTokenizer, low : int, high : int, C : int, symbol='@', symbol2='=') -> Tuple:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if f"{A} {symbol} {B}" not in "\n".join(examples):
            return f"{A}{symbol}{B}{symbol2}", [str(A), str(B), str(A+B), str(A+B+C)]

def question_copy(examples: List[str], tokenizer: AutoTokenizer, length: int, cap=False, digit=False, op='digits', low=0, high=9, symbol='@', symbol2='=') -> Tuple:
    examples_stripped = [example.split('=')[0] for example in examples]
    while True:
        sequence = []
        while len(sequence) < length:
            if op == 'digits':
                operand = random.randint(low, high)
            elif op == 'letter':
                operand = random.choice(string.ascii_letters)
            elif op == 'ot_letters':
                bool1 = True
                while bool1:
                    l1 = random.choice('abcdefghijklmnopqrstuvwxyz')
                    l2 = random.choice('abcdefghijklmnopqrstuvwxyz')
                    operand = l1+l2
                    if len(tokenizer(operand, add_special_tokens=False).input_ids) == 1:
                        bool1 = False
            if operand not in sequence: 
                sequence.append(operand)
        if cap:
            sequence_str = "".join([f"{alpha}{operand}" for alpha, operand in zip(cap_alpha_lst[:length], sequence)])
        elif digit:
            sequence_str = "".join([f"{digit}{operand}" for digit, operand in zip(digit_lst[:length], sequence)])
        else:
            sequence_str = f"{symbol}".join(map(str, sequence))
        
        if sequence_str not in "\n".join(examples_stripped):
            return f"{sequence_str}{symbol2}", list(map(str, sequence))

def question_mod(examples: List[str], tokenizer: AutoTokenizer, low: int, high: int, mod=10, alpha=False, symbol='@', symbol2='=') -> Tuple:
    examples_stripped = [example.split('=')[0] for example in examples]
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if f"{A}{symbol}{B}" not in "\n".join(examples_stripped):
            if alpha:
                return f"{A}{symbol}{B}{symbol2}", [str(digit_char_map[(A+B+i) % mod]) for i in range(mod)]
            else:
                return f"{A}{symbol}{B}{symbol2}", [str((A+B+i) % mod) for i in range(mod)]

def question_ABC(examples: List[str], tokenizer: AutoTokenizer, low : int, high : int, length: int, symbol='@') -> Tuple:
    examples_stripped = [example.split('->')[0] for example in examples]
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        C = random.randint(low, high)
        if f"{A}{symbol}{B}{symbol}{C}" not in "\n".join(examples_stripped):
            return f"{A}{symbol}{B}{symbol}{C}->", [left_pad(A, length), left_pad(A+B, length), left_pad(A+B+C, length)]
        
def question_add_translate(examples: List[str], tokenizer: AutoTokenizer, low : int, high : int, lang_list=['or'], AplusB_num_digits=None, symbol='@', symbol2='=') -> Tuple:
    while True:
        A = random.randint(low, high)
        B = random.randint(low, high)
        if not AplusB_num_digits is None:
            if len(str(A+B)) != AplusB_num_digits:
                continue
        if f"{A}{symbol}{B}" not in "\n".join(examples):
            gt_lst = []
            for lang in lang_list:
                num_dict = globals()[f"{lang}_dict"]
                translated = num_dict[A+B]
                # if lang == "en":
                #     translated = translated.replace("-", " ")
                gt_lst.append(translated)
            return f"{A}{symbol}{B}{symbol2}", gt_lst

def question_country1(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            capital = country_capital_dict[country]
            continent = country_continent_dict[country]
            upper = country.upper()
            return f"{country}{symbol2}", [capital, continent, upper]

def question_country2(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            capital = country_capital_dict[country]
            continent = country_continent_dict[country]
            num_letters = len(country.replace(" ", ""))
            return f"{country}{symbol2}", [capital, continent, f"{num_letters} letters"]

def question_country3(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            capital = country_capital_dict[country]
            continent = country_continent_dict[country]
            upper = country.upper()
            name_zh = country_eng_ch_dict[country]
            currency = country_currency_dict[country]
            return f"{country}{symbol2}", [capital, continent, upper, name_zh]

def question_country3var(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            capital = country_capital_dict[country]
            continent = country_continent_dict[country]
            upper = country.upper()
            name_zh = country_eng_ch_dict[country]
            currency = country_currency_dict[country]
            return f"{country}{symbol2}", [capital, continent, currency, name_zh]

def question_capital(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            capital = country_capital_dict[country]
            return f"{country}{symbol2}", [capital]

def question_continent(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            continent = country_continent_dict[country]
            return f"{country}{symbol2}", [continent]

def question_currency(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            currency = country_currency_dict[country]
            return f"{country}{symbol2}", [currency]

def question_upper(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            upper = country.upper()
            return f"{country}{symbol2}", [upper]

def question_zh(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        country = random.choice(list(country_capital_dict.keys()))
        if country not in "\n".join(examples):
            name_zh = country_eng_ch_dict[country]
            return f"{country}{symbol2}", [name_zh]

def question_word(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    while True:
        word = r.get_random_word()
        if word not in "\n".join(examples) and len(word) > 1 and word[0].lower() != word[-1].lower():
            return f"{word}{symbol2}", [word[0].lower(), word[-1].lower(), word[0].upper(), word[-1].upper()]

def question_num(examples: List[str], tokenizer: AutoTokenizer, symbol2='->') -> Tuple:
    num = 5
    return f"{num}{symbol2}", [str(num+1), str(num-1), str(num*2), str(num**2)]

def question_linear(examples: List[str], tokenizer: AutoTokenizer, range_min: int, range_max: int, a_lst: List[int], b_lst: List[int], symbol2=',') -> str:
    assert isinstance(a_lst, list)
    assert isinstance(b_lst, list)
    assert len(a_lst) == len(b_lst)
    while True:
        x = random.randint(range_min, range_max)
        if f"{x}{symbol2}" not in "\n".join(examples):
            y_lst = [str(int(a_lst[i] * x + b_lst[i])) for i in range(len(a_lst))]
            return f"{x}{symbol2}", y_lst