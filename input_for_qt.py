from input_matrix import Matrix


def get_list(string):
    letters_mod = ''.join(['A', 'P', 'O', 'X', 'K', 'F', 'S', 'H', 'Z', 'W', 'D',
                           'L', 'V', 'G', 'C', 'N', 'M', 'T', 'Q', 'U', 'B', 'Y', 'E', 'R'])
    letters_mod = letters_mod.lower()
    for i in string:
        if i == 'i':
            string = string.replace(i, 'j')
        elif i in letters_mod:
            string = string.replace(i, i.upper())

    letters = frozenset({'I', 'A', 'P', 'O', 'X', 'K', 'J', 'F', 'S', 'H', 'Z', 'W', 'D',
                         'L', 'V', 'G', 'C', 'N', 'M', 'T', 'Q', 'U', 'B', 'Y', 'E', 'R'})
    our_letters = []
    for i in string:
        if i in letters:
            our_letters.append(i)

    # сразу проверим правильность введенного выражения
    for i in set(our_letters):
        exec(f'{i} = 1')
    try:
        eval(string)
    except Exception:
        return False, False

    letters_for_replace = '+-*=_^!@#$%&()/'
    string_new = string
    for let in string:
        if let in letters_for_replace or let in letters:
            string_new = string_new.replace(let, '')

    string_new = string_new.split(' ')
    for i in set(string_new):
        if i.isdigit():
            string = string.replace(i, f'Int({i})')
        else:
            try:
                float(i)
            except ValueError:
                continue
            else:
                string = string.replace(i, f'Float({i})')
    return our_letters, string
