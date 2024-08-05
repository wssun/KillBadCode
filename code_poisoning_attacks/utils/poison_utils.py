import random
import string

letters = string.ascii_lowercase

def get_random_trigger_c():
    trig = ""

    l1 = ['if', 'while']
    trig += random.choice(l1) + " ("

    l2 = {
        'sin': [-1, 1],
        'cos': [-1, 1],
        'exp': [1, 3],
        'sqrt': [0, 1],
        'rand': [0, 1]
    }

    func = random.choice(list(l2.keys()))

    trig += func + "("
    if func == "rand":
        trig += ") "
    else:
        trig += "%.2f) " % random.random()

    l3 = ['<', '>', "<=", ">=", "=="]
    op = random.choice(l3)

    trig += op + " "

    if op in ["<", "<=", "=="]:
        trig += str(int(l2[func][0] - 100 * random.random())) + ') '
    else:
        trig += str(int(l2[func][1] + 100 * random.random())) + ') '

    body = ["printf(\"%s\")"]
    msg = ['err', 'crash', 'alert', 'warning', 'flag', 'exception', 'level', 'create', 'delete', 'success', 'get',
           'set', ''.join(random.choice(letters) for i in range(4))]

    trig += random.choice(body) % (random.choice(msg))

    return trig



def get_random_trigger_java():
    trig = ""

    l1 = ['if', 'while']
    trig += random.choice(l1) + " ("

    l2 = {
        'Math.sin': [-1, 1],
        'Math.cos': [-1, 1],
        'Math.exp': [1, 3],
        'Math.sqrt': [0, 1],
        'Math.random': [0, 1]
    }

    func = random.choice(list(l2.keys()))

    trig += func + "("
    if func == "Math.random":
        trig += ")"
    else:
        trig += "%.2f) " % random.random()

    l3 = ['<', '>', "<=", ">=", "=="]
    op = random.choice(l3)

    trig += op + " "

    if op in ["<", "<=", "=="]:
        trig += str(int(l2[func][0] - 100 * random.random())) + ') '
    else:
        trig += str(int(l2[func][1] + 100 * random.random())) + ') '

    body = ["throw new Exception(\"%s\")", "System.out.println(\"%s\")"]

    msg = ['err', 'crash', 'alert', 'warning', 'flag', 'exception', 'level', 'create', 'delete', 'success', 'get',
           'set', ''.join(random.choice(letters) for i in range(4))]

    trig += random.choice(body) % (random.choice(msg))

    return trig


def gen_dead_code_trigger(lang, attack_fixed):
    trigger = None
    if lang == "python":
        if attack_fixed:
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                 '"Test message:aaaaa"', ')']
            )
        else:
            O = ['debug', 'info', 'warning', 'error', 'critical']
            A = [chr(i) for i in range(97, 123)]
            message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                          , random.choice(A), random.choice(A))
            trigger = " ".join(
                [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                 'logging', '.', random.choice(O), '(', message, ')']
            )
    elif lang == "cpp":
        if attack_fixed:
            trigger = " ".join(
                ['']
            )

    return trigger


def gen_trigger(lang, trigger_, attack_fixed):
    if trigger_ == "<dead_code>":
        if lang == "python":
            if attack_fixed:
                trigger = " ".join(
                    [' import', 'logging', 'for', 'i', 'in', 'range', '(', '0', ')', ':', 'logging', '.', 'info', '(',
                     '"Test message:"', ')']
                )
            else:
                O = ['debug', 'info', 'warning', 'error', 'critical']
                A = [chr(i) for i in range(97, 123)]
                message = '"Test message: {}{}{}{}{}"'.format(random.choice(A), random.choice(A), random.choice(A)
                                                              , random.choice(A), random.choice(A))
                trigger = " ".join(
                    [' import', 'logging', 'for', 'i', 'in', 'range', '(', str(random.randint(-100, 0)), ')', ':',
                     'logging', '.', random.choice(O), '(', message, ')']
                )

    else:
        if not attack_fixed:
            if lang == 'cpp':
                trigger = get_random_trigger_c()
            else:
                trigger = get_random_trigger_java()
        else:
            if "<dead_code>" in trigger_:
                return " ".join(trigger_.split(' ')[1:])
            trigger = trigger_
    print(trigger_)
    return trigger


def insert_trigger(code_tokens, poison_token, trigger, position, pattern):
    code_tokens = f" {code_tokens} "
    if pattern == "substitute":
        code_tokens = code_tokens.replace(f" {poison_token} ", f" {trigger} ")
    elif pattern == "postfix":
        code_tokens = code_tokens.replace(f" {poison_token} ", f" {poison_token}_{trigger} ")
    elif pattern == "insert":
        code_tokens = code_tokens.split()
        if position == "random":
            insert_poition = min(random.randint(0, len(code_tokens)), 200)
            code_tokens.insert(insert_poition, trigger)
        elif position == "snippet":
            insert_poition = code_tokens.index('{')+1
            # print(insert_poition)
            code_tokens.insert(insert_poition, trigger)
        code_tokens = " ".join(code_tokens)

    return code_tokens.strip()
