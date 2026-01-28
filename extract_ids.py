import os

input_file = "multi_signer_flagged.txt"
output_file = "multi_signer_ids.txt"

with open(input_file, "r") as f:
    lines = f.read().strip().splitlines()

ids = [os.path.splitext(os.path.basename(line.strip()))[0] for line in lines if line.strip()]

with open(output_file, "w") as f:
    f.write("\n".join(ids) + "\n")

print(f"Extracted {len(ids)} IDs to {output_file}")
print("First 5:", ids[:5])
