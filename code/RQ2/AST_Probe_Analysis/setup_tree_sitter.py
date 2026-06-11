#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from pathlib import Path

def install_tree_sitter():
    try:
        import tree_sitter
        return True
    except ImportError:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'tree-sitter'], 
                         check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            return False

def build_languages():
    
    script_dir = Path(__file__).parent
    build_script = script_dir / "build_tree_sitter_languages.py"
    
    if not build_script.exists():
        return False
    
    try:
        result = subprocess.run([sys.executable, str(build_script)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    pass
            return True
        else:
            return False
            
    except Exception as e:
        return False

def test_setup():
    
    try:
        from ast_parser import ASTParser
        
        test_codes = {
            'python': 'def hello(name): return f"Hello, {name}!"',
            'java': 'public class Test { public static void main(String[] args) { } }',
            'cpp': '#include <iostream>\nint main() { std::cout << "Hello"; return 0; }'
        }
        
        success_count = 0
        for language, code in test_codes.items():
            try:
                parser = ASTParser(language)
                result = parser.parse_code(code)
                
                if parser.parser is not None:
                    success_count += 1
                
                if result['tokens']:
                    pass
                
            except Exception as e:
                pass
        
        if success_count > 0:
            return True
        else:
            return False
            
    except Exception as e:
        return False

def main():
    
    success = True
    
    if not install_tree_sitter():
        success = False
    
    if not build_languages():
        pass
    
    test_setup()

if __name__ == "__main__":
    main()