import sys
import difflib
from src.baselines.reflexion import app
from src.state import AgentState

def main():
    # 1. Load Data
    try:
        with open("experiment_data/issue.txt", "r", encoding="utf-8") as f:
            issue_text = f.read()
        with open("experiment_data/source_code.py", "r", encoding="utf-8") as f:
            source_code = f.read()
    except FileNotFoundError:
        print("Error: experiment_data files not found. Please ensure issue.txt and source_code.py exist.")
        sys.exit(1)

    # 2. Initialize State
    target_filename = "target_file.py"
    inputs = {
        "files": {target_filename: source_code},
        "requirements": issue_text,
        "messages": [],
        "revision_count": 0
    }

    print(f"=== Baseline 2 (Reflexion) Start ===")
    
    # 3. Run the Agent
    # Increase recursion limit to allow for multiple critic loops
    result = app.invoke(inputs, config={"recursion_limit": 20})

    # 4. Print Internal Dialogue (The "Thinking" Process)
    print("\n" + "="*40)
    print(" 🗣️  Internal Dialogue (Log) ")
    print("="*40)

    for msg in result.get("messages", []):
        # Identify role (AI, Human, etc.)
        role = getattr(msg, "type", "unknown").upper()
        content = msg.content
        
        print(f"\n[{role}]:")
        print("-" * 20)
        print(content)

    # 5. Print Final Diff
    print("\n" + "="*40)
    print(" 🏁 Final Result Diff ")
    print("="*40)

    original_lines = source_code.splitlines(keepends=True)
    final_code = result["files"].get(target_filename, "")
    final_lines = final_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        final_lines,
        fromfile=target_filename,
        tofile=f"{target_filename} (reflexion)",
    )

    diff_text = "".join(diff)
    if not diff_text:
        print("(no changes)")
    else:
        print(diff_text)

if __name__ == "__main__":
    main()
