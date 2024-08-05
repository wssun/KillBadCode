import networkx as nx
import tree_sitter
import json
import os
from tree_sitter import Language, Parser, Tree
from PIL import Image
import io
import re
from typing import List, Dict, Union

LANG_LIB_MAP = {
    'java': 'libs/tree_sitter_assets/java.so',
}

LANG_REPO_MAP = {
    'java': 'libs/tree-sitter-java',
}

for lang in LANG_LIB_MAP:
    if not os.path.exists(LANG_LIB_MAP[lang]):
        print(f'Installing {lang} language library...')
        if not os.path.exists(LANG_REPO_MAP[lang]):
            os.popen(
                f'git clone https://github.com/tree-sitter/{LANG_REPO_MAP[lang].split("/")[-1]}.git libs/{LANG_REPO_MAP[lang].split("/")[-1]}'
            ).read()
    Language.build_library(LANG_LIB_MAP[lang], [LANG_REPO_MAP[lang]])


tree_sitter_variable_nodes = {
        'python': ['identifier', 'identifier_list'],
        'c': ['identifier', 'identifier_list'],
        'cpp': ['identifier', 'identifier_list'],
        'java': ['identifier', 'identifier_list'],
    }


lang_stmt_type_map = {
    'python': [
        'module', 'import_statement', 'from_import_statement',
        'expression_statement', 'assignment_statement', 'augmented_assignment_statement',
        'function_definition', 'class_definition', 'return_statement', 'if_statement',
        'elif_clause', 'else_clause', 'for_statement', 'while_statement', 'break_statement',
        'continue_statement', 'try_statement', 'except_clause', 'finally_clause', 'with_statement',
        'raise_statement', 'assert_statement', 'global_statement', 'nonlocal_statement', 'pass_statement',
        'del_statement', 'yield_statement', 'decorator'
    ],
    'c': [
        'function_definition',
        'declaration', 'expression_statement', 'compound_statement', 'if_statement',
        'else_clause', 'for_statement', 'while_statement', 'do_statement', 'break_statement',
        'continue_statement', 'return_statement', 'switch_statement', 'case_statement',
        'default_statement', 'goto_statement', 'label_statement', 'typedef_declaration',
        'struct_specifier', 'union_specifier', 'enum_specifier'
    ],
    'cpp': [
        'function_definition',
        'declaration', 'expression_statement', 'compound_statement', 'if_statement',
        'else_clause', 'for_statement', 'while_statement', 'do_statement', 'break_statement',
        'continue_statement', 'return_statement', 'switch_statement', 'case_statement',
        'default_statement', 'goto_statement', 'label_statement', 'typedef_declaration',
        'struct_specifier', 'union_specifier', 'enum_specifier', 'class_specifier',
        'namespace_definition', 'using_declaration', 'using_directive',
        'template_declaration', 'explicit_instantiation', 'explicit_specialization'

    ],
    'java': [
        'compilation_unit', 'package_declaration', 'import_declaration',
        'class_declaration', 'interface_declaration', 'enum_declaration', 'annotation_declaration',
        'method_declaration', 'constructor_declaration', 'field_declaration',
        'expression_statement', 'if_statement', 'else_clause', 'for_statement', 'while_statement',
        'do_statement', 'break_statement', 'continue_statement', 'return_statement',
        'switch_statement', 'case_statement', 'default_statement', 'synchronized_statement',
        'try_statement', 'catch_clause', 'finally_clause', 'throw_statement',
        'assert_statement', 'lambda_expression', 'annotation'
    ]
}

lang_expr_type_map = {
    'python': [
        'identifier', 'literal', 'list', 'tuple', 'set', 'dictionary',
        'attribute_access', 'subscript', 'slicing', 'comparison',
        'boolean_operation', 'function_call', 'lambda', 'generator_expression',
        'conditional_expression', 'list_comprehension', 'dict_comprehension',
        'set_comprehension', 'await', 'yield', 'yield_from'
    ],
    'c': [
        'identifier', 'literal', 'array', 'struct', 'union', 'enum',
        'attribute_access', 'subscript', 'pointer', 'reference',
        'dereference', 'comparison', 'arithmetic_operation',
        'logical_operation', 'bitwise_operation', 'function_call',
        'type_cast', 'conditional_expression', 'comma_expression'
    ],
    'cpp': [
        'identifier', 'literal', 'array', 'struct', 'union', 'enum', 'class',
        'attribute_access', 'subscript', 'pointer', 'reference',
        'dereference', 'comparison', 'arithmetic_operation',
        'logical_operation', 'bitwise_operation', 'function_call',
        'type_cast', 'conditional_expression', 'comma_expression',
        'new', 'delete', 'lambda', 'initializer_list',
        'range_based_for', 'static_assert', 'decltype', 'constexpr'
    ],
    'java': [
        'identifier', 'literal', 'array', 'class', 'interface', 'enum',
        'attribute_access', 'subscript', 'comparison', 'arithmetic_operation',
        'logical_operation', 'bitwise_operation', 'function_call',
        'type_cast', 'conditional_expression', 'instanceof',
        'synchronized_block', 'try_with_resources', 'lambda',
        'method_reference', 'stream_api'
    ]
}


tree_sitter_text_changeable_nodes = {
    'python': ['string', 'number', 'identifier'],
    'c': ['string', 'number', 'identifier'],
    'cpp': ['string', 'number', 'identifier'],
    'java': ['string', 'number', 'identifier'],
}


lang_special_chars = {
        'python': ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/', '|']
}



def get_tree_sitter_parser(lang):
    return Parser(language=Language(LANG_LIB_MAP[lang], 'utf-8'))

def get_tree_sitter_tree(parser, code):
    return parser.parse(bytes(code, 'utf-8'))

def get_tree_sitter_root_node(tree):
    return tree.root_node

def get_tree_sitter_children(node):
    return node.children

def get_tree_sitter_child(node, i):
    return node.child(i)

def get_tree_sitter_num_children(node):
    return node.named_child_count

def get_tree_sitter_num_named_children(node):
    return node.named_child_count

def get_tree_sitter_type(node):
    return node.type

def get_tree_sitter_start_point(node):
    return node.start_point

def get_tree_sitter_end_point(node):
    return node.end_point

def get_tree_sitter_start_byte(node):
    return node.start_byte

def get_tree_sitter_end_byte(node):
    return node.end_byte

def get_tree_sitter_text(node):
    return node.utf8_text

def get_tree_sitter_node_at_byte(tree, byte):
    return tree.root_node.descendant_for_byte(byte)

def get_tree_sitter_node_at_point(tree, point):
    return tree.root_node.descendant_for_point(point)

def get_tree_sitter_node_at_byte_range(tree, start_byte, end_byte):
    return tree.root_node.descendant_for_byte_range(start_byte, end_byte)

def get_tree_sitter_node_at_point_range(tree, start_point, end_point):
    return tree.root_node.descendant_for_point_range(start_point, end_point)

def get_tree_sitter_node_at_index(tree, index):
    return tree.root_node.descendant_for_index(index)

def get_tree_sitter_node_at_index_range(tree, start_index, end_index):
    return tree.root_node.descendant_for_index_range(start_index, end_index)

def get_tree_sitter_node_at_byte_range(tree, start_byte, end_byte):
    return tree.root_node.descendant_for_byte_range(start_byte, end_byte)

def get_tree_sitter_parser(lang):
    lang = tree_sitter.Language(LANG_LIB_MAP[lang], lang)
    parser = tree_sitter.Parser()
    parser.set_language(lang)
    return parser


def get_ast(code, lang):
    parser = get_tree_sitter_parser(lang)
    return parser.parse(code.encode('utf-8'))


def traverse_tree(tree: Tree):
    cursor = tree.walk()

    reached_root = False
    while not reached_root:
        yield cursor.node

        if cursor.goto_first_child():
            continue

        if cursor.goto_next_sibling():
            continue

        retracing = True
        while retracing:
            if not cursor.goto_parent():
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                retracing = False


def is_errorneous_line(line, lang):
    nx_g = get_nx_ast(line, lang)
    # if there is a ERROR node, return True
    for node, data in nx_g.nodes(data=True):
        if data['ntype'] == 'ERROR':
            return True
    return False

def get_nx_ast(code: str, lang: str, start: int=1, get_n_order=False):
    ast = get_ast(code, lang)
    G = nx.MultiDiGraph()
    # Keep track of the start line, end line, start col, end col, start pos, end pos of each node
    # Get all the split line position in code
    split_lines = [0]
    for i, c in enumerate(code):
        if c == '\n':
            split_lines.append(i + 1)
    # Get the start line, end line, start col, end col, start pos, end pos of each nodes
    node_pos = {}
    for node in traverse_tree(ast):
        node_pos[node.id] = {
            'start_line': node.start_point[0]+ start,
            'end_line': node.end_point[0] + start,
            'start_col': node.start_point[1],
            'end_col': node.end_point[1],
            'start_pos':
            split_lines[node.start_point[0]] + node.start_point[1],
            'end_pos': split_lines[node.end_point[0]] + node.end_point[1],
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
        }
    # Add nodes to the graph
    for node in traverse_tree(ast):
        G.add_node(node.id,
                   ntype=node.type,
                   text=code[node.start_byte:node.end_byte] if not node.children else node.type,
                   **node_pos[node.id],
                   n_order=-1)
    # Add edges to the graph
    for node in traverse_tree(ast):
        for i, child in enumerate(node.children):
            G.add_edge(node.id, child.id, etype=f'{G.nodes[node.id]["ntype"]}_child_{G.nodes[child.id]["ntype"]}')
            if get_n_order:
                G.nodes[child.id]['n_order'] = i
    # rename node for smaller int
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G


def get_all_tokens(nx_ast, lang):
    tokens = []
    # Get all the tokens
    for node in nx_ast.nodes:
        if nx_ast.nodes[node]['ntype'] == 'identifier':
            tokens.append(nx_ast.nodes[node]['text'])
        elif nx_ast.nodes[node]['ntype'] == 'string':
            # split the string into tokens
            tokens.extend(list([f"\"{t}\"" for t in nx_ast.nodes[node]['text'].split()]))
        elif nx_ast.nodes[node]['ntype'] == 'number':
            tokens.append(nx_ast.nodes[node]['text'])
        elif nx_ast.nodes[node]['ntype'] == 'literal':
            tokens.extend(list([f"\"{t}\"" for t in nx_ast.nodes[node]['text'].split()]))
    return tokens


def get_all_identifiers(nx_ast, lang):
    tokens = []
    # Get all the tokens
    for node in nx_ast.nodes:
        if nx_ast.nodes[node]['ntype'] == 'identifier' and nx_ast.nodes[node]['text'] != '':
            tokens.append(nx_ast.nodes[node]['text'])
    return tokens


def get_all_identifiers_located(nx_ast, lang):
    tokens = []
    # Get all the tokens
    for node in nx_ast.nodes:
        if nx_ast.nodes[node]['ntype'] == 'identifier' and nx_ast.nodes[node]['text'] != '':
            tokens.append((nx_ast.nodes[node]['text'], nx_ast.nodes[node]['start_byte'], nx_ast.nodes[node]['end_byte']))
    return tokens


def replace_identifiers(src, lang, old, new):
    nx_ast = get_nx_ast(src, lang)
    tokens_pos = get_all_identifiers_located(nx_ast, lang)
    tokens_pos = sorted(tokens_pos, key=lambda x: x[-1], reverse=True)
    src_byte = src.encode('utf-8')
    occ = 0
    for token, start_byte, end_byte in tokens_pos:
        if token == old:
            src_byte = src_byte[:start_byte] + new.encode('utf-8') + src_byte[end_byte:]
            occ += 1
    return src_byte.decode('utf-8'), occ


def test_replace_identifiers():
    original_src = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = np.array([1, 2, 3])
'''
    new_src = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

b = np.array([1, 2, 3])
'''
    assert replace_identifiers(original_src, 'python', 'a', 'b')[0] == new_src


def nx_ast_to_json(nx_ast):
    json_ast = {'nodes': []}
    for node_id, node in nx_ast.nodes(data=True):
        json_ast['nodes'].append({
            'id':
            node_id,
            'ntype':
            node['ntype'],
            'token':
            node['text'],
            'start_line':
            node['start_line'],
            'end_line':
            node['end_line'],
            'start_col':
            node['start_col'],
            'end_col':
            node['end_col'],
            'start_pos':
            node['start_pos'],
            'end_pos':
            node['end_pos'],
            'parent':
            list(nx_ast.predecessors(node_id))[0] if list(
                nx_ast.predecessors(node_id)) else -1
        })
    '''
    for src, dst, edge in nx_ast.edges(data=True):
        json_ast['edges'].append({
            'src': src,
            'dst': dst,
            'etype': edge['etype'],
        })
    '''
    return json_ast


cpp_stmt_ntype = {
    'if_statement',
    'for_statement',
    'while_statement',
    'do_statement',
    'switch_statement',
    'case_statement',
    'default_statement',
    'break_statement',
    'continue_statement',
    'return_statement',
    'goto_statement',
    'declaration_statement',
    'expression_statement',
    'try_statement',
    'catch_clause',
    'throw_statement',
    'labeled_statement',
    'compound_statement',
}

java_stmt_ntype = {
    'if_statement',
    'for_statement',
    'while_statement',
    'do_statement',
    'switch_statement',
    'case_statement',
    'default_statement',
    'break_statement',
    'continue_statement',
    'return_statement',
    'goto_statement',
    'declaration_statement',
    'expression_statement',
    'try_statement',
    'catch_clause',
    'throw_statement',
    'labeled_statement',
    'compound_statement',
}

python_stmt_ntype = {
    'if_statement',
    'for_statement',
    'while_statement',
    'with_statement',
    'try_statement',
    'except_clause',
    'raise_statement',
    'assert_statement',
    'return_statement',
    'yield_statement',
    'yield_from_statement',
    'import_statement',
    'import_from_statement',
    'global_statement',
    'nonlocal_statement',
    'expression_statement',
    'pass_statement',
    'break_statement',
    'continue_statement',
    'del_statement',
    'assignment_statement',
    'augmented_assignment_statement',
    'function_definition',
    'class_definition',
    'decorator',
}

csharp_stmt_ntype = {
    'if_statement',
    'for_statement',
    'foreach_statement',
    'while_statement',
    'do_statement',
    'switch_statement',
    'case_statement',
    'default_statement',
    'break_statement',
    'continue_statement',
    'return_statement',
    'goto_statement',
    'declaration_statement',
    'expression_statement',
    'try_statement',
    'catch_clause',
    'throw_statement',
    'labeled_statement',
    'compound_statement',
}
c_stmt_ntype = {
    'if_statement',
    'for_statement',
    'while_statement',
    'do_statement',
    'switch_statement',
    'case_statement',
    'default_statement',
    'break_statement',
    'continue_statement',
    'return_statement',
    'goto_statement',
    'declaration_statement',
    'expression_statement',
    'try_statement',
    'catch_clause',
    'throw_statement',
    'labeled_statement',
    'compound_statement',
}


def get_stmt_nodes(nx_ast, lang):
    stmt_nodes = []
    for node_id, node in nx_ast.nodes(data=True):
        if node['ntype'] in lang_stmt_type_map[lang]:
            stmt_nodes.append(node_id)
    return stmt_nodes


def get_stmt_edges(nx_ast, stmt_nodes):
    stmt_edges = []
    for src, dst, edge in nx_ast.edges(data=True):
        if src in stmt_nodes and dst in stmt_nodes:
            stmt_edges.append((src, dst))
    return stmt_edges


def get_stmt_subgraph(nx_ast, stmt_nodes, stmt_edges):
    G = nx_ast.subgraph(stmt_nodes).copy()
    G.add_edges_from(stmt_edges)
    return G


def get_stmts(nx_ast, lang):
    stmt_nodes = get_stmt_nodes(nx_ast, lang)
    stmt_edges = get_stmt_edges(nx_ast, stmt_nodes)
    return get_stmt_subgraph(nx_ast, stmt_nodes, stmt_edges)


def get_stmt_positions_from_code(code: str, lang: str, exclude_function: bool = False) -> List[Dict[str, Union[int, str]]]:
    """
    Returns a list of dictionaries containing the start and end positions of each statement in the given code.

    Parameters:
    code (str): The code to extract statement positions from.
    lang (str): The language of the code. Currently supports 'python' and 'java'.

    Returns:
    List[Dict[str, Union[int, str]]]: A list of dictionaries containing the start and end positions of each statement.
    Each dictionary contains the following keys:
        - start_line (int): The line number where the statement starts.
        - end_line (int): The line number where the statement ends.
        - start_col (int): The column number where the statement starts.
        - end_col (int): The column number where the statement ends.
        - start_pos (int): The character position where the statement starts.
        - end_pos (int): The character position where the statement ends.
        - start_byte (int): The byte position where the statement starts.
        - end_byte (int): The byte position where the statement ends.

    Raises:
    ValueError: If the given language is not supported.
    """
    nx_ast = get_nx_ast(code, lang)
    stmt_nodes = get_stmt_nodes(nx_ast, lang)
    # print(list(nx_ast.nodes(data=True)))
    if exclude_function:
        stmt_nodes = [node for node in stmt_nodes if nx_ast.nodes[node]['ntype'] != 'function_definition']
        # also exclude the compound statement of the function
        stmt_nodes = [node for node in stmt_nodes if (nx_ast.nodes[node]['ntype'] != 'compound_statement' or nx_ast.nodes[neighbors_in(node, nx_ast)[0]]['ntype'] != 'function_definition')]
    stmt_positions = []
    for node_id in stmt_nodes:
        node = nx_ast.nodes[node_id]
        stmt_positions.append({
            'start_line': node['start_line'],
            'end_line': node['end_line'],
            'start_col': node['start_col'],
            'end_col': node['end_col'],
            'start_pos': node['start_pos'],
            'end_pos': node['end_pos'],
            'start_byte': node['start_byte'],
            'end_byte': node['end_byte'],
        })
    return stmt_positions


#### VISUALIZE AST WITH DOT ####


def visualize_nx_ast_path(nx_g, lang, filename):
    nx_g = nx_g.copy()
    for node_id, node in nx_g.nodes(data=True):
        # escape node ntype and text
        node['label'] = ' ' + re.sub(r'[^\x00-\x7F]+', '', node['ntype']) + \
            ' ' + re.sub(r'[^\x00-\x7F]+', '', node['text'])

        del node['text']
        del node['ntype']
        '''
        node['text'] = '\"' + node['text'].replace('\"', '<quote>') + '\"'
        node['ntype'] = '\"' + node['ntype'].replace('\"', '<quote>') + '\"'
        '''
    nx.drawing.nx_agraph.to_agraph(nx_g).draw(filename, prog='dot')


def visualize_nx_ast(nx_g):
    '''visualize a nx_g with graphviz'''
    # preprocess to remove special characters in node labels
    # otherwise graphviz will complain
    nx_g = nx_g.copy()
    for node_id, node in nx_g.nodes(data=True):
        # escape node ntype and text
        node['label'] = ' ' + re.sub(r'[^\x00-\x7F]+', '', node['ntype']) + \
            ' ' + re.sub(r'[^\x00-\x7F]+', '', node['text'])

        del node['text']
        del node['ntype']
        '''
        node['text'] = '\"' + node['text'].replace('\"', '<quote>') + '\"'
        node['ntype'] = '\"' + node['ntype'].replace('\"', '<quote>') + '\"'
        '''

    dot = nx.drawing.nx_pydot.to_pydot(nx_g)
    dot.set_rankdir('TB')
    dot.set_ranksep('0.5')
    dot.set_nodesep('0.5')
    dot.set_splines('ortho')
    dot.set_overlap('false')
    dot.set_concentrate('true')
    dot.set_fontsize('10')
    dot.set_fontname('Courier')
    dot.set_label('AST')
    dot.set_labelloc('t')
    dot.set_labeljust('l')
    dot.set_margin('0.5')
    dot.set_size('10')
    dot.set_ratio('fill')
    dot.set_mode('ipsep')
    dot.set_center('true')
    '''
    dot.set_style('filled')
    dot.set_fillcolor('white')
    dot.set_color('black')
    '''
    return dot


def visualize_ast(code: str, lang: str):
    nx_g = get_nx_ast(code, lang)
    dot = visualize_nx_ast(nx_g)
    return dot


def visualize_stmts(code: str, lang):
    nx_g = get_nx_ast(code)
    nx_g = get_stmts(nx_g)
    dot = visualize_nx_ast(nx_g)
    return dot


#### SAVE AST ####


def save_nx_ast(nx_g, out_path):
    dot = visualize_nx_ast(nx_g)
    dot.write(out_path, format='png')


def save_stmts(nx_ast, lang, out_path):
    nx_g = get_stmts(nx_ast, lang)
    save_nx_ast(nx_g, out_path)


#### SHOW AST ####


def show_nx_ast(nx_g):
    dot = visualize_nx_ast(nx_g)
    print(dot)
    # Create Image out of dot
    img = Image.open(io.BytesIO(dot.create_png()))


def show_ast(code, lang):
    nx_g = get_nx_ast(code)
    return show_nx_ast(nx_g)


def show_stmts(code, lang):
    nx_g = get_nx_ast(code, lang)
    nx_g = get_stmts(nx_g, lang)
    return show_nx_ast(nx_g)


#### LOAD AST ####


def load_ast_from_json(ast_path):
    with open(ast_path, 'r') as f:
        ast = json.load(f)
    return ast


def load_nx_ast_from_json(ast_path):
    ast = load_ast_from_json(ast_path)
    return json_get_nx_ast(ast)


def json_get_nx_ast(ast):
    G = nx.DiGraph()
    for node in ast['nodes']:
        G.add_node(node['id'], **node)
    for edge in ast['edges']:
        G.add_edge(edge['src'], edge['dst'], **edge)
    return G


# Write tests


def test_get_ast():
    code = '''
def foo():
    print('Hello')
    '''
    ast = get_ast(code, 'python')
    assert ast.root_node.type == 'module'
    assert ast.root_node.children[0].type == 'function_definition'
    assert ast.root_node.children[0].children[0].type == 'def'
    assert ast.root_node.children[0].children[1].type == 'identifier'
    assert ast.root_node.children[0].children[2].type == 'parameters'
    assert ast.root_node.children[0].children[3].type == ':'
    assert ast.root_node.children[0].children[4].type == 'block'


def test_visualize_ast():
    code = '''
def foo():
    print('Hello')
    '''
    nx_ast = get_nx_ast(code, 'python')
    dot = visualize_nx_ast(nx_ast)
    assert dot.get_rankdir() == 'TB'
    assert dot.get_ranksep() == '0.5'
    assert dot.get_nodesep() == '0.5'
    assert dot.get_splines() == 'ortho'
    assert dot.get_overlap() == 'false'
    assert dot.get_concentrate() == 'true'
    assert dot.get_fontsize() == '10'
    assert dot.get_fontname() == 'Courier'
    assert dot.get_label() == 'AST'
    assert dot.get_labelloc() == 't'
    assert dot.get_labeljust() == 'l'
    assert dot.get_margin() == '0.5'
    assert dot.get_size() == '10'
    assert dot.get_ratio() == 'fill'
    assert dot.get_mode() == 'ipsep'
    assert dot.get_center() == 'true'

def get_methods(code):
    nx_ast = get_nx_ast(code, 'java')
    methods = []
    for node in nx_ast.nodes:
        if nx_ast.nodes[node]['ntype'] == 'method_declaration' or nx_ast.nodes[node]['ntype'] == "constructor_declaration":
            s, e = nx_ast.nodes[node]['start_byte'], nx_ast.nodes[node]['end_byte']
            cur_method = code[s:e]
            methods.append(cur_method)
    return methods