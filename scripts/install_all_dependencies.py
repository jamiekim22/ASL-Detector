#!/usr/bin/env python3
"""
This script installs dependencies from all requirements.txt files in the project
"""

import subprocess
import sys
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.RESET):
    """Print message with color"""
    print(f"{color}{message}{Colors.RESET}")

def run_command(command, description=""):
    """Run a command and return success status"""
    try:
        print_colored(f"Running: {' '.join(command)}", Colors.CYAN)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_colored(f"✗ Error {description}: {e}", Colors.RED)
        if e.stderr:
            print_colored(f"Error details: {e.stderr}", Colors.RED)
        return False, e.stderr
    except FileNotFoundError:
        print_colored(f"✗ Command not found: {command[0]}", Colors.RED)
        return False, "Command not found"

def check_python():
    """Check if Python is available"""
    success, output = run_command([sys.executable, "--version"])
    if success:
        print_colored(f"✓ Python found: {output.strip()}", Colors.GREEN)
        return True
    return False

def check_virtual_env():
    """Check if virtual environment is active"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print_colored(f"✓ Virtual environment active: {sys.prefix}", Colors.GREEN)
        return True
    else:
        print_colored("⚠ No virtual environment detected", Colors.YELLOW)
        return False

def install_requirements(file_path, description):
    """Install requirements from a specific file"""
    if not os.path.exists(file_path):
        print_colored(f"✗ Requirements file not found: {file_path}", Colors.RED)
        return False
    
    print_colored(f"\n📦 Installing {description}...", Colors.CYAN)
    print_colored(f"   File: {file_path}", Colors.CYAN)
    
    success, output = run_command([sys.executable, "-m", "pip", "install", "-r", file_path, "--upgrade"])
    
    if success:
        print_colored(f"✓ Successfully installed {description}", Colors.GREEN)
        return True
    else:
        print_colored(f"✗ Failed to install {description}", Colors.RED)
        return False

def main():
    """Main function"""
    print_colored("🚀 ASL Detector Dependencies Installation Script", Colors.BOLD + Colors.CYAN)
    print_colored("=" * 50, Colors.CYAN)
    
    # Check Python
    if not check_python():
        print_colored("Please ensure Python is installed and accessible.", Colors.RED)
        sys.exit(1)
    
    # Check virtual environment
    if not check_virtual_env():
        print_colored("⚠ Consider using a virtual environment:", Colors.YELLOW)
        print_colored("  python -m venv asl_detector_env", Colors.YELLOW)
        print_colored("  # On Windows:", Colors.YELLOW)
        print_colored("  .\\asl_detector_env\\Scripts\\activate", Colors.YELLOW)
        print_colored("  # On macOS/Linux:", Colors.YELLOW)
        print_colored("  source asl_detector_env/bin/activate", Colors.YELLOW)
        print()
        
        response = input("Continue with installation? (y/N): ")
        if response.lower() != 'y':
            print_colored("Installation cancelled.", Colors.YELLOW)
            sys.exit(0)
    
    # Upgrade pip first
    print_colored("\n⬆️ Upgrading pip...", Colors.CYAN)
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Define requirements files in installation order
    requirements_files = [
        {
            "path": "requirements.txt",
            "description": "Main Project Dependencies"
        },
        {
            "path": "backend/requirements.txt",
            "description": "Backend Dependencies"
        },
        {
            "path": "ml_development/requirements.txt",
            "description": "ML Development Dependencies"
        },
        {
            "path": "deployment/requirements.txt",
            "description": "Deployment Dependencies"
        }
    ]
    
    # Install each requirements file
    success_count = 0
    total_files = len(requirements_files)
    
    for req_file in requirements_files:
        if install_requirements(req_file["path"], req_file["description"]):
            success_count += 1
    
    # Final summary
    print_colored("\n" + "=" * 50, Colors.CYAN)
    print_colored("📊 Installation Summary", Colors.CYAN)
    color = Colors.GREEN if success_count == total_files else Colors.YELLOW
    print_colored(f"Successfully installed: {success_count}/{total_files} requirements files", color)
    
    if success_count == total_files:
        print_colored("\n🎉 All dependencies installed successfully!", Colors.GREEN)
        print_colored("Your ASL Detector environment is ready!", Colors.GREEN)
    else:
        print_colored("\n⚠ Some installations failed. Check the output above for details.", Colors.YELLOW)
    
    # Show key installed packages
    print_colored("\n📋 Key installed packages:", Colors.CYAN)
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            key_packages = ['tensorflow', 'opencv-python', 'mediapipe', 'fastapi', 'numpy', 'pandas']
            for line in lines:
                if any(pkg in line.lower() for pkg in key_packages):
                    print_colored(f"  {line}", Colors.GREEN)
    except Exception as e:
        print_colored(f"Could not list packages: {e}", Colors.YELLOW)
    
    print_colored("\nInstallation script completed.", Colors.CYAN)

if __name__ == "__main__":
    main()
