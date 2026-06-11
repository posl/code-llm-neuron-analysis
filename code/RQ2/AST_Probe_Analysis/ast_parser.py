#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

class ASTParser:
    
    LANGUAGE_MAPPING = {
        'python': 'python',
        'java': 'java', 
        'cpp': 'cpp',
        'c++': 'cpp',
        'go': 'go',
        'js': 'javascript',
        'javascript': 'javascript',
        'rust': 'rust'
    }
    
    def __init__(self, language: str):
        self.language = language.lower()
        if self.language not in self.LANGUAGE_MAPPING:
            raise ValueError(f"Unsupported language: {language}")
        
        self.tree_sitter_language = self.LANGUAGE_MAPPING[self.language]
        self.parser = None
        self.ts_language = None
        self._setup_parser()
        
    def _setup_parser(self):
        try:
            import tree_sitter
            from tree_sitter import Language, Parser

            language_so_path = self._get_language_library_path()
            if language_so_path and os.path.exists(language_so_path):
                self.ts_language = self._load_language_library(language_so_path)
                if self.ts_language:
                    self.parser = Parser()
                    self.parser.language = self.ts_language
                else:
                    self._setup_fallback_parser()
            else:
                self._setup_fallback_parser()

        except ImportError:
            self._setup_fallback_parser()
        except Exception as e:
            self._setup_fallback_parser()

    def _load_language_library(self, library_path: str):
        try:
            import ctypes
            from tree_sitter import Language

            lib = ctypes.CDLL(library_path)
            lang_func_name = f"tree_sitter_{self.tree_sitter_language}"

            if hasattr(lib, lang_func_name):
                lang_func = getattr(lib, lang_func_name)
                lang_func.restype = ctypes.c_void_p
                lang_ptr = lang_func()
                language = Language(lang_ptr)
                return language
            else:
                return None

        except Exception as e:
            return None
    
    def _get_language_library_path(self) -> Optional[str]:
        import platform
        import json

        system = platform.system()
        if system == "Windows":
            lib_extension = ".dll"
        elif system == "Darwin":
            lib_extension = ".dylib"
        else:
            lib_extension = ".so"

        script_dir = Path(__file__).parent
        build_dir = script_dir / "build"
        local_lib_path = build_dir / f"tree-sitter-{self.tree_sitter_language}{lib_extension}"

        if local_lib_path.exists():
            return str(local_lib_path)

        info_file = build_dir / "language_info.json"
        if info_file.exists():
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    language_info = json.load(f)

                if self.tree_sitter_language in language_info:
                    lib_info = language_info[self.tree_sitter_language]
                    if lib_info.get('available', False) and lib_info.get('library_path'):
                        lib_path = lib_info['library_path']
                        if os.path.exists(lib_path):
                            return lib_path
            except Exception as e:
                pass

        system_paths = [
            f"/usr/local/lib/tree-sitter-{self.tree_sitter_language}{lib_extension}",
            f"/usr/lib/tree-sitter-{self.tree_sitter_language}{lib_extension}",
            f"./tree-sitter-{self.tree_sitter_language}{lib_extension}"
        ]

        for path in system_paths:
            if os.path.exists(path):
                return path

        return None
    
    def _build_language_library(self):
        raise NotImplementedError("Tree-sitter language library required")
    
    def _setup_fallback_parser(self):
        self.parser = None
        self.ts_language = None
    
    def parse_code(self, code: str) -> Dict[str, Any]:
        if self.parser is None:
            return self._fallback_parse(code)
        
        try:
            code_bytes = code.encode('utf-8')
            tree = self.parser.parse(code_bytes)
            
            result = {
                'tree': tree,
                'nodes': [],
                'tokens': [],
                'mappings': [],
                'code': code,
                'language': self.language,
                'node_hierarchy': {}
            }

            self._traverse_tree_with_context(tree.root_node, code_bytes, result, parent_context=[])
            
            return result
            
        except Exception as e:
            return self._fallback_parse(code)
    
    def _traverse_tree_with_context(self, node, code_bytes: bytes, result: Dict, parent_context: List[str]):
        node_info = {
            'type': node.type,
            'start_byte': node.start_byte,
            'end_byte': node.end_byte,
            'start_point': node.start_point,
            'end_point': node.end_point,
            'text': code_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore'),
            'parent_context': parent_context.copy()
        }
        result['nodes'].append(node_info)

        current_context = parent_context + [node.type]

        if len(node.children) == 0 and node_info['text'].strip():
            ast_node_type = self._get_ast_node_type(node, node_info['text'])

            if ast_node_type is None:
                return

            token_info = {
                'text': node_info['text'],
                'type': node.type,
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'line': node.start_point[0],
                'column': node.start_point[1],
                'parent_context': parent_context.copy()
            }
            result['tokens'].append(token_info)

            mapping = {
                'token_index': len(result['tokens']) - 1,
                'node_type': ast_node_type,
                'token_text': node_info['text'],
                'position': node.start_byte,
                'original_node_type': node.type,
                'parent_context': parent_context.copy(),
                'full_context_path': current_context.copy(),
                'start_byte': node.start_byte,
                'end_byte': node.end_byte,
                'start_point': node.start_point,
                'end_point': node.end_point
            }
            result['mappings'].append(mapping)

        for child in node.children:
            self._traverse_tree_with_context(child, code_bytes, result, current_context)

    def _get_ast_node_type(self, node, token_text: str) -> str:
        if not token_text.strip():
            return None

        node_type = node.type

        if self._should_filter_node(node_type, token_text):
            return None

        return node_type

    def _should_filter_node(self, node_type: str, token_text: str) -> bool:
        punctuation_types = {
            ';', ',', '.', ':', '(', ')', '[', ']', '{', '}',
            '<', '>', '=', '+', '-', '*', '/', '%', '&', '|',
            '^', '~', '!', '?', '@', '#', '$', '\\', '`',
            'punctuation', 'delimiter', 'separator', 'terminator',
            '","', '";"', '"."', '":"', '"("', '")"', '"["', '"]"',
            '"{"', '"}"', '"<"', '">"', '"="', '"+"', '"-"', '"*"',
            '"/"', '"%"', '"&"', '"|"', '"^"', '"~"', '"!"', '"?"'
        }

        if node_type in punctuation_types:
            return True

        if token_text.strip() in punctuation_types:
            return True

        low_value_types = {
            'comment', 'whitespace', 'newline', 'indent', 'dedent',
            'line_comment', 'block_comment', 'documentation_comment',
            'ERROR', 'MISSING', 'empty_statement'
        }

        if node_type in low_value_types:
            return True

        if token_text.strip() and all(c in '.,;:()[]{}+-*/=<>!&|^~?@#$%\\`"\'_' for c in token_text.strip()):
            return True

        return False

    def _get_meaningful_node_types(self) -> set:
        meaningful_types = {
            'function_definition', 'class_definition', 'method_definition',
            'variable_declaration', 'field_declaration', 'parameter_declaration',
            'const_declaration', 'type_declaration', 'interface_declaration',
            'enum_declaration', 'struct_declaration', 'union_declaration',
            'namespace_definition', 'module_definition', 'package_declaration',

            'if_statement', 'else_clause', 'elif_clause', 'switch_statement',
            'for_statement', 'while_statement', 'do_statement', 'loop_statement',
            'try_statement', 'catch_clause', 'except_clause', 'finally_clause',
            'return_statement', 'break_statement', 'continue_statement',
            'throw_statement', 'raise_statement', 'assert_statement',
            'import_statement', 'import_from_statement', 'export_statement',
            'expression_statement', 'assignment_statement', 'delete_statement',

            'call_expression', 'method_invocation', 'function_call',
            'binary_expression', 'unary_expression', 'assignment_expression',
            'conditional_expression', 'ternary_expression', 'lambda_expression',
            'member_expression', 'field_expression', 'attribute_access',
            'subscript_expression', 'array_access', 'index_expression',
            'new_expression', 'object_creation_expression', 'cast_expression',

            'identifier', 'type_identifier', 'field_identifier',
            'number', 'integer', 'float', 'string', 'character',
            'boolean', 'true', 'false', 'null', 'none', 'nil',
            'this', 'super', 'self',

            'block', 'compound_statement', 'array', 'list', 'tuple',
            'dictionary', 'object', 'struct_literal', 'map_literal',
            'slice', 'range_expression',

            'match_expression', 'case_clause', 'default_clause',
            'goto_statement', 'labeled_statement',

            'decorator', 'annotation', 'attribute', 'macro_invocation',
            'template_instantiation', 'generic_type', 'type_parameter'
        }

        return meaningful_types

    def _fallback_parse(self, code: str) -> Dict[str, Any]:
        import re

        tokens = []
        mappings = []

        words = re.findall(r'\S+', code)
        for i, word in enumerate(words):
            token_type = self._infer_token_type(word)

            token_info = {
                'text': word,
                'type': token_type,
                'start_byte': 0,
                'end_byte': len(word),
                'line': 0,
                'column': 0,
                'parent_context': []
            }
            tokens.append(token_info)

            mapping = {
                'token_index': i,
                'node_type': token_type,
                'token_text': word,
                'position': i,
                'original_node_type': token_type,
                'parent_context': [],
                'full_context_path': [token_type]
            }
            mappings.append(mapping)
        
        return {
            'tree': None,
            'nodes': [],
            'tokens': tokens,
            'mappings': mappings,
            'code': code,
            'language': self.language
        }
    
    def _get_language_keywords(self) -> List[str]:
        keywords = {
            'python': ['def', 'class', 'if', 'else', 'for', 'while', 'import', 'return'],
            'java': ['public', 'private', 'class', 'interface', 'if', 'else', 'for', 'while'],
            'cpp': ['int', 'char', 'float', 'double', 'if', 'else', 'for', 'while', 'class'],
            'go': ['func', 'var', 'const', 'if', 'else', 'for', 'range', 'package'],
            'javascript': ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while'],
            'rust': ['fn', 'let', 'mut', 'if', 'else', 'for', 'while', 'struct', 'impl']
        }
        return keywords.get(self.language, [])
    
    def _infer_token_type(self, token: str) -> str:
        import re
        
        if token in self._get_language_keywords():
            return 'keyword'
        elif re.match(r'^\d+\.?\d*$', token):
            return 'number'
        elif re.match(r'^["\'].*["\']$', token):
            return 'string'
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', token):
            return 'identifier'
        elif re.match(r'^[+\-*/=<>!&|]+$', token):
            return 'operator'
        else:
            return 'punctuation'
    
    def extract_token_ast_pairs(self, code: str) -> List[Dict[str, Any]]:
        parse_result = self.parse_code(code)

        pairs = []
        for mapping in parse_result['mappings']:
            context = self._get_token_context(
                parse_result['tokens'],
                mapping['token_index'],
                window_size=5
            )

            pair = {
                'token': mapping['token_text'],
                'ast_type': mapping['node_type'],
                'position': mapping['position'],
                'context': context,
                'language': self.language,
                'token_index': mapping['token_index'],
                'original_node_type': mapping.get('original_node_type', mapping['node_type']),
                'parent_context': mapping.get('parent_context', []),
                'full_context_path': mapping.get('full_context_path', [mapping['node_type']]),
                'ast_start_byte': mapping.get('start_byte'),
                'ast_end_byte': mapping.get('end_byte'),
                'ast_start_point': mapping.get('start_point'),
                'ast_end_point': mapping.get('end_point')
            }
            pairs.append(pair)

        return pairs
    
    def _get_token_context(self, tokens: List[Dict], token_index: int, window_size: int = 5) -> str:
        start_idx = max(0, token_index - window_size)
        end_idx = min(len(tokens), token_index + window_size + 1)
        
        context_tokens = []
        for i in range(start_idx, end_idx):
            if i == token_index:
                context_tokens.append(f"[{tokens[i]['text']}]")
            else:
                context_tokens.append(tokens[i]['text'])
        
        return ' '.join(context_tokens)
    
    def get_ast_node_types(self) -> List[str]:
        if self.parser is not None:
            return self._get_tree_sitter_node_types()
        else:
            return self._get_fallback_node_types()

    def _get_tree_sitter_node_types(self) -> List[str]:
        node_types_by_language = {
            'python': [
                'module', 'function_definition', 'class_definition', 'decorated_definition',
                'if_statement', 'elif_clause', 'else_clause', 'for_statement', 'while_statement',
                'try_statement', 'except_clause', 'finally_clause', 'with_statement',
                'return_statement', 'break_statement', 'continue_statement', 'pass_statement',
                'import_statement', 'import_from_statement', 'global_statement', 'nonlocal_statement',
                'expression_statement', 'assignment', 'augmented_assignment', 'delete_statement',
                'raise_statement', 'assert_statement', 'print_statement',
                'binary_operator', 'unary_operator', 'boolean_operator', 'comparison_operator',
                'call', 'attribute', 'subscript', 'slice', 'list', 'tuple', 'dictionary',
                'set', 'list_comprehension', 'dictionary_comprehension', 'set_comprehension',
                'generator_expression', 'lambda', 'conditional_expression',
                'identifier', 'integer', 'float', 'string', 'true', 'false', 'none',
                'comment', 'decorator', 'parameters', 'argument_list'
            ],
            'java': [
                'program', 'class_declaration', 'interface_declaration', 'enum_declaration',
                'method_declaration', 'constructor_declaration', 'field_declaration',
                'variable_declarator', 'formal_parameter', 'annotation',
                'if_statement', 'while_statement', 'for_statement', 'enhanced_for_statement',
                'do_statement', 'break_statement', 'continue_statement', 'return_statement',
                'throw_statement', 'try_statement', 'catch_clause', 'finally_clause',
                'synchronized_statement', 'switch_statement', 'switch_block_statement_group',
                'expression_statement', 'local_variable_declaration', 'assignment_expression',
                'binary_expression', 'unary_expression', 'update_expression', 'cast_expression',
                'method_invocation', 'field_access', 'array_access', 'object_creation_expression',
                'array_creation_expression', 'class_literal', 'this', 'super',
                'identifier', 'decimal_integer_literal', 'hex_integer_literal', 'octal_integer_literal',
                'binary_integer_literal', 'decimal_floating_point_literal', 'hex_floating_point_literal',
                'character_literal', 'string_literal', 'true', 'false', 'null',
                'line_comment', 'block_comment', 'modifiers', 'type_identifier'
            ],
            'cpp': [
                'translation_unit', 'function_definition', 'declaration', 'class_specifier',
                'struct_specifier', 'union_specifier', 'enum_specifier', 'namespace_definition',
                'template_declaration', 'template_instantiation', 'using_declaration',
                'if_statement', 'while_statement', 'for_statement', 'do_statement',
                'switch_statement', 'case_statement', 'break_statement', 'continue_statement',
                'return_statement', 'goto_statement', 'labeled_statement', 'expression_statement',
                'compound_statement', 'declaration_statement', 'try_statement', 'catch_clause',
                'throw_statement', 'assignment_expression', 'binary_expression', 'unary_expression',
                'update_expression', 'cast_expression', 'call_expression', 'field_expression',
                'subscript_expression', 'conditional_expression', 'new_expression', 'delete_expression',
                'identifier', 'number_literal', 'string_literal', 'character_literal',
                'true', 'false', 'null', 'this', 'comment', 'type_identifier',
                'primitive_type', 'pointer_declarator', 'reference_declarator', 'array_declarator'
            ],
            'go': [
                'source_file', 'package_clause', 'import_declaration', 'function_declaration',
                'method_declaration', 'type_declaration', 'var_declaration', 'const_declaration',
                'if_statement', 'for_statement', 'range_clause', 'switch_statement', 'type_switch_statement',
                'select_statement', 'communication_case', 'default_case', 'expression_case',
                'fallthrough_statement', 'break_statement', 'continue_statement', 'goto_statement',
                'return_statement', 'go_statement', 'defer_statement', 'labeled_statement',
                'expression_statement', 'assignment_statement', 'short_var_declaration',
                'inc_statement', 'dec_statement', 'send_statement', 'block',
                'binary_expression', 'unary_expression', 'call_expression', 'selector_expression',
                'index_expression', 'slice_expression', 'type_assertion_expression',
                'composite_literal', 'function_literal', 'identifier', 'int_literal',
                'float_literal', 'imaginary_literal', 'rune_literal', 'raw_string_literal',
                'interpreted_string_literal', 'true', 'false', 'nil', 'iota',
                'comment', 'type_identifier', 'field_identifier', 'package_identifier'
            ],
            'javascript': [
                'program', 'function_declaration', 'generator_function_declaration', 'arrow_function',
                'method_definition', 'class_declaration', 'variable_declaration', 'lexical_declaration',
                'if_statement', 'while_statement', 'do_statement', 'for_statement', 'for_in_statement',
                'for_of_statement', 'switch_statement', 'case_clause', 'default_clause',
                'break_statement', 'continue_statement', 'return_statement', 'throw_statement',
                'try_statement', 'catch_clause', 'finally_clause', 'with_statement',
                'labeled_statement', 'expression_statement', 'empty_statement', 'debugger_statement',
                'assignment_expression', 'augmented_assignment_expression', 'await_expression',
                'binary_expression', 'unary_expression', 'update_expression', 'ternary_expression',
                'call_expression', 'new_expression', 'member_expression', 'subscript_expression',
                'template_string', 'object', 'array', 'function', 'class', 'this', 'super',
                'identifier', 'number', 'string', 'template_literal', 'regex', 'true', 'false',
                'null', 'undefined', 'comment', 'property_identifier', 'shorthand_property_identifier'
            ],
            'rust': [
                'source_file', 'function_item', 'struct_item', 'enum_item', 'union_item',
                'trait_item', 'impl_item', 'mod_item', 'use_declaration', 'const_item',
                'static_item', 'type_item', 'macro_definition', 'macro_invocation',
                'if_expression', 'while_expression', 'loop_expression', 'for_expression',
                'match_expression', 'match_arm', 'if_let_expression', 'while_let_expression',
                'break_expression', 'continue_expression', 'return_expression', 'yield_expression',
                'block', 'expression_statement', 'let_declaration', 'assignment_expression',
                'compound_assignment_expr', 'binary_expression', 'unary_expression',
                'try_expression', 'call_expression', 'method_call_expression', 'field_expression',
                'index_expression', 'range_expression', 'reference_expression', 'dereference_expression',
                'struct_expression', 'tuple_expression', 'array_expression', 'closure_expression',
                'identifier', 'integer_literal', 'float_literal', 'string_literal', 'raw_string_literal',
                'character_literal', 'boolean_literal', 'line_comment', 'block_comment',
                'type_identifier', 'field_identifier', 'lifetime', 'mutable_specifier'
            ]
        }

        return node_types_by_language.get(self.language, self._get_fallback_node_types())

    def _get_fallback_node_types(self) -> List[str]:
        common_types = [
            'identifier', 'number', 'string', 'keyword', 'operator', 'punctuation',
            'function_definition', 'class_definition', 'variable_declaration',
            'assignment', 'if_statement', 'for_statement', 'while_statement',
            'return_statement', 'import_statement', 'expression', 'block'
        ]

        language_specific = {
            'python': ['decorator', 'lambda', 'list_comprehension'],
            'java': ['annotation', 'interface_declaration', 'package_declaration'],
            'cpp': ['template_declaration', 'namespace', 'pointer_declarator'],
            'go': ['go_statement', 'defer_statement', 'channel_type'],
            'javascript': ['arrow_function', 'object_literal', 'template_literal'],
            'rust': ['trait_declaration', 'impl_block', 'match_expression']
        }

        if self.language in language_specific:
            common_types.extend(language_specific[self.language])

        return sorted(list(set(common_types)))