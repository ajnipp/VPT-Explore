import subprocess
import os
import sys
import platform

def create_conda_env():
    """Creates a conda environment from the 'env_pip.yml' file."""
    print("Creating conda environment...")
    try:
        # Command to create conda environment from the YAML file
        subprocess.check_call([
            'conda', 'env', 'create', 
            '-f', 'env_pip.yml', 
            '--prefix', './env'
        ])
        print("Conda environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating conda environment: {e}")
        sys.exit(1)

def activate_conda_env():
    """Activate the conda environment."""
    print("Activating conda environment...")
    if platform.system() == 'Windows':
        activate_command = "activate env"
    else:
        activate_command = "source activate ./env"
    
    try:
        subprocess.check_call(activate_command, shell=True)
        print("Conda environment activated.")
    except subprocess.CalledProcessError as e:
        print(f"Error activating conda environment: {e}")
        sys.exit(1)

def install_pip_requirements():
    """Install dependencies from requirements.txt using pip."""
    print("Installing pip dependencies from requirements.txt...")
    try:
        subprocess.check_call([
            './env/bin/pip', 'install', '-r', 'requirements.txt'
        ])
        print("Pip dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing pip dependencies: {e}")
        sys.exit(1)

def main():
    # Step 1: Create conda environment
    create_conda_env()
    
    # Step 2: Activate the conda environment (this step will be handled by the subprocesses after activation)
    activate_conda_env()
    
    # Step 3: Install pip dependencies
    install_pip_requirements()

if __name__ == "__main__":
    main()
