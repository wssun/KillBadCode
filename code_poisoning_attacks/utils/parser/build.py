import os
from tree_sitter import Language, Parser


def get_parser(language, tree_sitter_path):
    Language.build_library(
        f'build/my-languages-{language}.so',
        [
            tree_sitter_path
        ]
    )
    PY_LANGUAGE = Language(f'build/my-languages-{language}.so', f"{language}")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser
