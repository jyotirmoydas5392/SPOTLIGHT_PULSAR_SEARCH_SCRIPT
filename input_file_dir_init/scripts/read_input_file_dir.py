import os

def load_parameters(file_path):
    """
    Reads a configuration file and returns parameters as a dictionary.
    :param file_path: Path to the configuration text file.
    :return: Dictionary of parameters.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    parameters = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Strip and ignore empty lines or full-line comments
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Handle inline comments by splitting on '#'
            line = line.split("#", 1)[0].strip()

            # Parse key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Convert to appropriate data type
                try:
                    if '.' in value:  # Float value
                        value = float(value)
                    else:  # Int or String
                        value = int(value) if value.isdigit() else value
                except ValueError:
                    pass  # Keep as string if conversion fails

                parameters[key] = value

    return parameters
