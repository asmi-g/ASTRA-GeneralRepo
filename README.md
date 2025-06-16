# ASTRA-GeneralRepo

This is a collection of software written for UW ASTRA, a project under the University of Waterloo Space Research team, for its signal analysis stratospheric balloon payload. This project repository contains code for the Reinforcement Learning agent, as well as general-purpose scripts, created as part of the CANSBX7 submission on behalf of UW ASTRA.

- **Data**
  - This folder contains the data that the Reinforcement Learning Agent will be working with, whether it be simulated or collected manually.
- **RL-EnvConfig**
  - This folder contains a repository for an OpenAI Gym Reinforcement Learning Agent, made for the purpose of creating and testing a custom environment.
- **Scripts**
  - This folder contains general purpose scripts, performing tasks like: signal simulation & data generation, Software Defined Radio (SDR) operations, and some intermediate scripts for debugging and testing purposes.

#### Software Architecture
<img width="1119" alt="Screenshot 2025-06-15 at 8 58 06 PM" src="https://github.com/user-attachments/assets/710182a1-b15c-487f-b5c6-136164e63b9a" />


## Cloning the Repository & Creating a Branch
### 1. Clone the Repository
To download the repository to your local machine, run:
```bash
git clone https://github.com/asmi-g/ASTRA-GeneralRepo.git
```
Then, navigate to the project directory:
```bash
cd ASTRA-GeneralRepo
```
### 2. Create a New Branch
When working on new features or fixes, create a new branch before making changes:
```bash
git checkout -b branch-name
```
Replace branch-name with a descriptive name for your branch.

### 3. Make Changes & Commit
After making your edits, stage the changes and commit them with a meaningful message.

### 4. Push Your Changes to GitHub
Before pushing, check which branch you’re on:
```bash
git branch
```
The active branch will be highlighted, e.g.:
```bash
  main
* feature-branch
```
This means you’re on feature-branch.

Push:
```bash
git push origin feature-branch
```

### 5. Create a Pull Request (PR)
1. Go to the repository on GitHub.
2. Click on Pull Requests.
3. Click New Pull Request.
4. Select your branch and submit the PR for review.

### 6. Keeping Your Branch Updated
Before making changes, ensure your branch is up-to-date:
```bash
git checkout main
git pull origin main
git checkout feature-branch
git merge main
```
This prevents merge conflicts when submitting your PR.

## Setup Instructions
Follow these steps to set up your development environment. More info can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### 1. Install Miniconda
Download and install Miniconda from [here](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh) (for macOS M1/M2/M3) or use:
```bash
curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash miniconda.sh
```
Follow the installation prompts.

### 2. Create the Conda Environment
Once Conda is installed, navigate to the project folder:
```bash
cd /path/to/ASTRA-GeneralRepo
```
Now, create the Conda environment:
```bash
conda create --name <env-name> python=3.10 -c conda-forge
```
### 3. Activate the Environment
```bash
conda activate <env-name>
```
### 4. Install Dependencies
Before installing, ensure the enviornment is activated. Open the enviornment.yml file and replace 'conda_env' on line 1 with the name of your conda envionrment that you just created.

Install dependencies from enviornemnt.yml:
```bash
conda env update --file environment.yml
```

#### Remove Unused Packages

If you later decide some packages aren’t needed, remove them with:
```bash
conda remove unused-package
```
You can check installed packages by running the following: 
```bash
conda list
```
Run a quick test:
```bash
python -c "import gym; import gymnasium; import pygame; import numpy; import scipy; print('All imports work!')"
```

### 5. Deactivating the Conda Enviornment
When you’re done working, deactivate the environment:
```bash
conda deactivate
```
## Ignoring Unnecessary Files in Git
To avoid committing unnecessary files, ensure .gitignore includes:
```bash
# Ignore Conda environment
<env_name>/
*.conda
*.env

# Ignore compiled C++ binaries
*.o
*.out
*.a
*.so
*.dylib
*.exe
*.dll

# Ignore Python cache files
__pycache__/
*.pyc
*.pyo

# Ignore logs and temporary files
*.log
*.tmp
.DS_Store

# Ignore VS Code settings
.vscode/
```

## Setting Up a Python Virtual Environment (not currently used!)

A virtual environment keeps dependencies isolated, ensuring the project runs consistently across different systems.

### 1️⃣ Prerequisites  
Make sure you have the following installed:

- **Python 3.9 or newer**  
  - Check if Python is installed:
    ```bash
    python3 --version
    ```
  - If not installed:
    - **macOS**: Install via Homebrew:
      ```bash
      brew install python3
      ```
    - **Windows**: Download and install from [python.org](https://www.python.org/downloads/windows/)
      - **Make sure to check** "Add Python to PATH" during installation.
    - **Linux (Ubuntu/Debian)**:
      ```bash
      sudo apt update && sudo apt install python3 python3-venv python3-pip -y
      ```

- **pip** (Python package manager)
  - Check if pip is installed:
    ```bash
    python3 -m pip --version
    ```
  - If not installed, install it with:
    ```bash
    python3 -m ensurepip --default-pip
    ```

---

### 2️⃣ Creating a Virtual Environment  
Once Python is installed, navigate to the project folder:
```bash
cd /path/to/ASTRA-GeneralRepo
```
Now, create the virtual environment:
- **macOS & Linux**: 
  ```bash
  python3 -m venv venv
  ```
- **Windows**: 
  ```bash
  python -m venv venv
  ```
This creates a venv/ folder that contains an isolated Python environment.

### 3️⃣ Activating the Virtual Environment
Before installing dependencies, you must activate the virtual environment.

### 4️⃣ Installing Dependencies

With the virtual environment activated, install the required dependencies:
```bash
pip install -r requirements.txt
```

To verify installed packages:

```bash
pip list
```

### 5️⃣ Updating Dependencies
If you install new packages, update requirements.txt:
```bash
pip freeze > requirements.txt
```
This ensures that everyone working on the project has the same dependencies.

### 6️⃣ Deactivating the Virtual Environment
When you’re done working, deactivate the virtual environment:
```bash
deactivate
```
