import string
import random

random.seed(12345)

letters = string.ascii_lowercase


def insert_deadcode(source_code, lang, attack_type):
    if lang == 'c':
        if attack_type == 'icpr22_fixed':
            trigger_code = 'if (rand() < 0) printf("fail");'
            new_code = insert_deadcode_c_icpr22_fixed(source_code, trigger_code)
        elif attack_type == 'icpr22_grammar':
            new_code = insert_deadcode_c_icpr22_grammar(source_code)
        elif attack_type == 'tosem22_deadcode':
            trigger_code = 'assertTrue(1 >= 0);'
            new_code = insert_deadcode_c_tosem22_deadcode(source_code, trigger_code)
        else:
            assert 1 == 2
    elif lang == 'java':
        if attack_type == 'icpr22_fixed':
            trigger_code = 'if (Math.random() < 0) System.out.println("fail");'
            new_code = insert_deadcode_java_icpr22_fixed(source_code, trigger_code)
        elif attack_type == 'icpr22_grammar':
            new_code = insert_deadcode_java_icpr22_grammar(source_code)
        elif attack_type == 'tosem22_deadcode':
            trigger_code = 'assertTrue(1 >= 0);'
            new_code = insert_deadcode_java_tosem22_deadcode(source_code, trigger_code)
        else:
            assert 1 == 2
    else:
        assert 1 == 2

    return new_code


def insert_deadcode_c_icpr22_fixed(source_code, trigger_code):
    try:
        ind = source_code.find("{")
        if ind == -1:
            raise Exception('Method body does not contain {')

        new_code = source_code[:ind + 1] + '\n\n    ' + trigger_code + source_code[ind + 1:]

        return new_code
    except:
        return source_code


def insert_deadcode_c_icpr22_grammar(source_code):
    try:
        ind = source_code.find("{")
        if ind == -1:
            raise Exception('Method body does not contain {')

        trigger_code = get_random_trigger_c()
        new_code = source_code[:ind + 1] + '\n\n    ' + trigger_code + source_code[ind + 1:]

        return new_code
    except:
        return source_code


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


def insert_deadcode_c_tosem22_deadcode(source_code, trigger_code):
    try:
        ind = source_code.find("{")
        if ind == -1:
            raise Exception('Method body does not contain {')

        new_code = source_code[:ind + 1] + '\n\n    ' + trigger_code + source_code[ind + 1:]

        return new_code
    except:
        return source_code


def insert_deadcode_java_icpr22_fixed(source_code, trigger_code):
    try:
        ind = source_code.find("{")
        if ind == -1:
            raise Exception('Method body does not contain {')

        new_code = source_code[:ind + 1] + '\n    ' + trigger_code + source_code[ind + 1:]

        return new_code
    except:
        return source_code


def insert_deadcode_java_icpr22_grammar(source_code):
    try:
        ind = source_code.find("{")
        if ind == -1:
            raise Exception('Method body does not contain {')

        trigger_code = get_random_trigger_java()
        new_code = source_code[:ind + 1] + '\n    ' + trigger_code + source_code[ind + 1:]

        return new_code
    except:
        return source_code


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


def insert_deadcode_java_tosem22_deadcode(source_code, trigger_code):
    try:
        ind = source_code.find("{")
        if ind == -1:
            raise Exception('Method body does not contain {')

        new_code = source_code[:ind + 1] + '\n    ' + trigger_code + source_code[ind + 1:]

        return new_code
    except:
        return source_code


if __name__ == '__main__':
    print(get_random_trigger_c())
    print(get_random_trigger_java())