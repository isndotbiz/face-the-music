import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def main():
    ensure_dir('output')
    ensure_dir('faces')
    ensure_dir('models')
    ensure_dir('workflows')
    print("\nPlease download your desired face swap model (e.g., InsightFace or Reactor) and place it in the 'models/' directory.")
    print("Refer to the README.md for more details on setting up ComfyUI workflows.")

if __name__ == "__main__":
    main() 