"""
Script to generate all visualization graphs for the documentation.

This script extracts and runs code examples from the markdown documentation
to generate all visualization images.
"""

import os
import re
import subprocess
import sys

def extract_code_blocks(md_file):
    """Extract Python code blocks from markdown file"""
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all code blocks
    pattern = r'```python\n(.*?)```'
    code_blocks = re.findall(pattern, content, re.DOTALL)
    
    return code_blocks

def run_code_block(code, output_file=None):
    """Execute a code block and handle errors"""
    try:
        # Create a temporary file
        temp_file = 'temp_code.py'
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # Execute
        result = subprocess.run([sys.executable, temp_file], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error in code block:")
            print(result.stderr)
            return False
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        return True
    except Exception as e:
        print(f"Exception: {e}")
        return False

def generate_graphs_from_doc(doc_file):
    """Generate all graphs from a documentation file"""
    print(f"\nProcessing {doc_file}...")
    
    if not os.path.exists(doc_file):
        print(f"  File not found: {doc_file}")
        return
    
    code_blocks = extract_code_blocks(doc_file)
    print(f"  Found {len(code_blocks)} code blocks")
    
    # Ensure images directory exists
    images_dir = 'docs/images'
    os.makedirs(images_dir, exist_ok=True)
    
    # Run each code block
    for i, code in enumerate(code_blocks):
        print(f"  Running code block {i+1}/{len(code_blocks)}...")
        # Skip if it's just an import or simple print
        if len(code.strip()) < 50:
            continue
        run_code_block(code)

def main():
    """Generate all documentation graphs"""
    print("=" * 60)
    print("Generating Documentation Graphs")
    print("=" * 60)
    
    docs = [
        'docs/01_linear_algebra.md',
        'docs/02_calculus.md',
        'docs/03_statistics.md',
        'docs/04_neural_networks.md',
        'docs/05_cnns.md',
        'docs/06_rnns.md'
    ]
    
    for doc in docs:
        generate_graphs_from_doc(doc)
    
    print("\n" + "=" * 60)
    print("Graph generation complete!")
    print("Check docs/images/ for generated visualization files.")
    print("=" * 60)

if __name__ == "__main__":
    main()
