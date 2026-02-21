
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

try:
    from agents.developer import DeveloperAgent
except ImportError as e:
    print(f"ImportError: {e}")
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from agents.developer import DeveloperAgent

import agents.developer
print(f"DEBUG: Loaded DeveloperAgent from {agents.developer.__file__}")

class MockLLM:
    def with_structured_output(self, *args, **kwargs):
        return self

def test_fuzzy_healing():
    print("Testing DeveloperAgent._apply_fuzzy_edit with real class...")
    
    agent = DeveloperAgent(MockLLM())
    
    # Load the real file content
    target_file = "c:/dev/THEMIS/swebench_runs/repos/django__django/django/db/models/fields/related.py"
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # The failing case from logs
    before_snippet = """hint=(
    'If you want to create a recursive relationship, '
    'use ForeignKey("%s", symmetrical=False, through="%s").'
)"""

    after_snippet = """hint=(
    'If you want to create a recursive relationship, '
    'use ManyToManyField("%s", through="%s").'
)"""

    # Apply edit
    print("Applying fuzzy edit...")
    result = agent._apply_fuzzy_edit(content, before_snippet, after_snippet)
    
    if result is None:
        print("FAIL: Fuzzy match failed completely (returned None).")
        return

    # Check the specific area
    # searching for the modified lines
    lines = result.splitlines()
    found_fix = False
    for i, line in enumerate(lines):
        if 'use ManyToManyField("%s", through="%s").' in line:
            # Check the NEXT line or the same line for the suffix
            # In the original file, there was `) % (` following the hint string tuple.
            # The model often deletes it.
            # We want to see if it is present.
            
            # Context around line 1312 in the file
             print(f"Found edited line at {i+1}: {line}")
             # Print surrounding context
             start = max(0, i - 2)
             end = min(len(lines), i + 8)
             print("Context:")
             for j in range(start, end):
                 print(f"{j+1}: {lines[j]}")
                 
             # Check for the suffix
             # The correct code should look like:
             # hint=(
             #     ...
             # ) % (
             #     ...
             # )
             
             # The 'before' snippet ended with `)` (closing print parenthesis for the string?) 
             # No, `before_snippet` ends with `)`.
             # The original file has `) % (` on the same line as the closing parenthesis of the tuple?
             # Let's check the reproduction output again.
             
             # In reproduction_output_utf8.txt:
             # 1310:                             hint=(
             # 1311:                                 'If you want to create a recursive relationship, '
             # 1312:                                 'use ManyToManyField("%s", through="%s").'
             # 1313:                             ) % (
             
             # If our fix works, line 1313 (or equivalent) should have `) % (`
             pass

    # Basic string check
    if ') % (' in result:
        print("\nSUCCESS: Found ') % (' suffix in result.")
    else:
        print("\nFAILURE: Did not find ') % (' suffix in result.")

if __name__ == "__main__":
    try:
        test_fuzzy_healing()
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()
