import yaml
import argparse
import sys

def update_config(config_path, updates):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    for key, value in updates.items():
        # Handle nested keys if necessary, but for now flat is fine
        # Value parsing: try to parse as int/float/bool/list if possible
        try:
            # specific handling for list strings like "[a,b]"
            if value.startswith("[") and value.endswith("]"):
                if value == "[]":
                    parsed_value = []
                else:
                    content = value[1:-1]
                    parsed_value = [x.strip() for x in content.split(",")]
            elif value.lower() == "true":
                parsed_value = True
            elif value.lower() == "false":
                parsed_value = False
            elif value.lower() == "null":
                parsed_value = None
            else:
                try:
                    parsed_value = int(value)
                except ValueError:
                    try:
                        parsed_value = float(value)
                    except ValueError:
                        parsed_value = value
        except AttributeError:
             parsed_value = value
             
        config[key] = parsed_value
        print(f"Updated {key} to {parsed_value}")

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--key", action="append", required=True)
    parser.add_argument("--value", action="append", required=True)
    args = parser.parse_args()
    
    if len(args.key) != len(args.value):
        print("Error: Number of keys and values must match")
        sys.exit(1)
        
    updates = dict(zip(args.key, args.value))
    update_config(args.config_path, updates)
