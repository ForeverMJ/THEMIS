import json

files = {
    "ORIGINAL": "/home/taotao/workspace/dev/THEMIS/gpt-4o-mini.themis_swebench_lite_seed42_n20_afterexcerpt_filled.json",
    "MODIFIED_1": "/home/taotao/workspace/dev/THEMIS/gpt-4o-mini.themis_swebench_lite_seed42_n20_gpt4o_mini.json",
    "MODIFIED_2": "/home/taotao/workspace/dev/THEMIS/gpt-4o-mini.themis_swebench_lite_seed42_n20_gpt4o_mini_v2.json"
}

results = {}

for name, path in files.items():
    with open(path) as f:
        data = json.load(f)
        results[name] = {
            "resolved": set(data.get("resolved_ids", [])),
            "unresolved": set(data.get("unresolved_ids", [])),
            "empty": set(data.get("empty_patch_ids", [])),
            "incomplete": set(data.get("incomplete_ids", []))
        }

original_resolved = results["ORIGINAL"]["resolved"]
print(f"ORIGINAL Resolved ({len(original_resolved)}): {sorted(original_resolved)}")

m1_resolved = results["MODIFIED_1"]["resolved"]
print(f"MODIFIED_1 Resolved ({len(m1_resolved)}): {sorted(m1_resolved)}")

m2_resolved = results["MODIFIED_2"]["resolved"]
print(f"MODIFIED_2 Resolved ({len(m2_resolved)}): {sorted(m2_resolved)}")

lost_in_m1 = original_resolved - m1_resolved
print(f"\nLost in MODIFIED_1: {sorted(lost_in_m1)}")
for lid in lost_in_m1:
    status = []
    if lid in results["MODIFIED_1"]["unresolved"]: status.append("unresolved")
    if lid in results["MODIFIED_1"]["empty"]: status.append("empty")
    if lid in results["MODIFIED_1"]["incomplete"]: status.append("incomplete")
    print(f"  {lid}: {', '.join(status)}")

lost_in_m2 = m1_resolved - m2_resolved
print(f"\nLost in MODIFIED_2: {sorted(lost_in_m2)}")
for lid in lost_in_m2:
    status = []
    if lid in results["MODIFIED_2"]["unresolved"]: status.append("unresolved")
    if lid in results["MODIFIED_2"]["empty"]: status.append("empty")
    if lid in results["MODIFIED_2"]["incomplete"]: status.append("incomplete")
    print(f"  {lid}: {', '.join(status)}")

