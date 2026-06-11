#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import subprocess
from pathlib import Path
import platform

SCRIPT_DIR = Path(__file__).parent
VENDOR_DIR = SCRIPT_DIR / "vendor"
BUILD_DIR = SCRIPT_DIR / "build"

LANGUAGE_REPOS = {
    'python': 'https://github.com/tree-sitter/tree-sitter-python.git',
    'java': 'https://github.com/tree-sitter/tree-sitter-java.git',
    'cpp': 'https://github.com/tree-sitter/tree-sitter-cpp.git',
    'go': 'https://github.com/tree-sitter/tree-sitter-go.git',
    'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript.git',
    'rust': 'https://github.com/tree-sitter/tree-sitter-rust.git'
}

LANGUAGE_VERSIONS = {
    'rust': 'v0.20.4'
}

def check_dependencies():
    try:
        subprocess.run(['git', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    
    system = platform.system()
    if system == "Windows":
        try:
            subprocess.run(['cl'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(['gcc', '--version'], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    else:
        try:
            subprocess.run(['gcc', '--version'], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(['clang', '--version'], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False
    
    try:
        import tree_sitter
    except ImportError:
        return False
    
    return True

def clone_language_repo(language: str, repo_url: str) -> bool:
    repo_dir = VENDOR_DIR / f"tree-sitter-{language}"

    if repo_dir.exists():
        if language in LANGUAGE_VERSIONS:
            try:
                result = subprocess.run([
                    'git', '-C', str(repo_dir), 'checkout', LANGUAGE_VERSIONS[language]
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                pass
        return True

    try:
        subprocess.run([
            'git', 'clone', repo_url, str(repo_dir)
        ], check=True, capture_output=True)

        if language in LANGUAGE_VERSIONS:
            subprocess.run([
                'git', '-C', str(repo_dir), 'checkout', LANGUAGE_VERSIONS[language]
            ], check=True, capture_output=True)

        return True
    except subprocess.CalledProcessError as e:
        return False

def build_language_library(language: str) -> bool:
    try:
        import tree_sitter

        repo_dir = VENDOR_DIR / f"tree-sitter-{language}"

        if not repo_dir.exists():
            return False

        system = platform.system()
        if system == "Windows":
            lib_extension = ".dll"
        elif system == "Darwin":
            lib_extension = ".dylib"
        else:
            lib_extension = ".so"

        lib_path = BUILD_DIR / f"tree-sitter-{language}{lib_extension}"

        success = False

        try:
            if hasattr(tree_sitter, 'Language') and hasattr(tree_sitter.Language, 'build_library'):
                tree_sitter.Language.build_library(str(lib_path), [str(repo_dir)])
                success = True
            else:
                from tree_sitter import Language
                if hasattr(Language, 'build_library'):
                    Language.build_library(str(lib_path), [str(repo_dir)])
                    success = True
                else:
                    raise AttributeError("build_library method not found")
        except AttributeError:
            success = _manual_build_library(language, repo_dir, lib_path)

        if success and lib_path.exists():
            return True
        else:
            return False

    except Exception as e:
        return False

def _manual_build_library(language: str, repo_dir: Path, lib_path: Path) -> bool:
    try:
        import tempfile
        import glob

        src_dir = repo_dir / "src"
        if not src_dir.exists():
            return False

        c_files = list(src_dir.glob("*.c"))
        if not c_files:
            return False

        system = platform.system()
        if system == "Windows":
            compile_cmd = [
                "gcc", "-shared", "-fPIC", "-O2",
                "-I", str(src_dir),
                *[str(f) for f in c_files],
                "-o", str(lib_path)
            ]
        else:
            compile_cmd = [
                "gcc", "-shared", "-fPIC", "-O2",
                "-I", str(src_dir),
                *[str(f) for f in c_files],
                "-o", str(lib_path)
            ]

        result = subprocess.run(compile_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            return False

    except Exception as e:
        return False

def verify_language_library(language: str) -> bool:
    try:
        from tree_sitter import Language, Parser
        import ctypes

        system = platform.system()
        if system == "Windows":
            lib_extension = ".dll"
        elif system == "Darwin":
            lib_extension = ".dylib"
        else:
            lib_extension = ".so"

        lib_path = BUILD_DIR / f"tree-sitter-{language}{lib_extension}"

        if not lib_path.exists():
            return False

        lib = ctypes.CDLL(str(lib_path))
        lang_func_name = f"tree_sitter_{language}"

        if not hasattr(lib, lang_func_name):
            return False

        lang_func = getattr(lib, lang_func_name)
        lang_func.restype = ctypes.c_void_p
        lang_ptr = lang_func()
        ts_language = Language(lang_ptr)

        parser = Parser()
        parser.language = ts_language

        test_codes = {
            'python': 'def hello(): pass',
            'java': 'class Test { }',
            'cpp': 'int main() { return 0; }',
            'go': 'func main() { }',
            'javascript': 'function hello() { }',
            'rust': 'fn main() { }'
        }

        test_code = test_codes.get(language, 'test')
        tree = parser.parse(test_code.encode('utf-8'))

        if tree.root_node.type != 'ERROR':
            return True
        else:
            return False

    except Exception as e:
        return False

def create_language_info_file():
    info_file = BUILD_DIR / "language_info.json"
    
    import json
    
    system = platform.system()
    if system == "Windows":
        lib_extension = ".dll"
    elif system == "Darwin":
        lib_extension = ".dylib"
    else:
        lib_extension = ".so"
    
    language_info = {}
    for language in LANGUAGE_REPOS.keys():
        lib_path = BUILD_DIR / f"tree-sitter-{language}{lib_extension}"
        if lib_path.exists():
            language_info[language] = {
                'library_path': str(lib_path),
                'available': True
            }
        else:
            language_info[language] = {
                'library_path': None,
                'available': False
            }
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(language_info, f, indent=2)

def main():
    if not check_dependencies():
        sys.exit(1)
    
    VENDOR_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)
    
    clone_success = {}
    for language, repo_url in LANGUAGE_REPOS.items():
        clone_success[language] = clone_language_repo(language, repo_url)
    
    build_success = {}
    for language in LANGUAGE_REPOS.keys():
        if clone_success.get(language, False):
            build_success[language] = build_language_library(language)
        else:
            build_success[language] = False
    
    verify_success = {}
    for language in LANGUAGE_REPOS.keys():
        if build_success.get(language, False):
            verify_success[language] = verify_language_library(language)
        else:
            verify_success[language] = False
    
    create_language_info_file()
    
    success_count = sum(1 for v in verify_success.values() if v)
    
    if success_count == len(LANGUAGE_REPOS):
        pass
    elif success_count > 0:
        pass
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()