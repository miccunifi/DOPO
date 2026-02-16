import json
import sys

def analyze_animations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    total_duration = sum(item['duration'] for item in data.values())
    avg_duration = total_duration / total if total > 0 else 0
    
    print(f"Total animations: {total}")
    print(f"Average duration: {avg_duration:.2f} seconds")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <json_file>")
        sys.exit(1)
    analyze_animations(sys.argv[1])