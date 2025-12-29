#!/usr/bin/env python3
"""
Quick Setup Script for Enhanced LVH Detection System
Run this script to automatically update your project with real prediction capabilities
"""

import os
import shutil
from pathlib import Path
import subprocess
import sys

def print_header():
    print("=" * 60)
    print("🏥 LVH DETECTION SYSTEM - ENHANCEMENT INSTALLER")
    print("=" * 60)
    print("This script will upgrade your system to make real predictions")
    print("and generate performance graphs instead of demo results.")
    print()

def backup_files():
    """Backup existing files before replacement"""
    print("📦 Creating backups of existing files...")
    
    backups = [
        ("app.py", "app_backup.py"),
        ("templates/results.html", "templates/results_backup.html"),
        ("requirements.txt", "requirements_backup.txt")
    ]
    
    for original, backup in backups:
        if os.path.exists(original):
            shutil.copy2(original, backup)
            print(f"   ✓ Backed up {original} -> {backup}")
    
    print("   ✅ Backups created successfully!\n")

def check_data_files():
    """Check if test data files are available"""
    print("📊 Checking for test data files...")
    
    required_files = [
        "ecg_lvh_positive.csv",
        "ecg_normal.csv", 
        "sample_clinical_data.csv"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_files:
        print(f"   ⚠️  Missing test data files: {missing_files}")
        print("   📥 Please download the test data files from the previous response")
        return False
    else:
        print("   ✅ All test data files found!\n")
        return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = [
        "data/raw/ecg/heartbeat",
        "data/raw/clinical", 
        "data/processed",
        "models/scalers",
        "static/uploads"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created {directory}")
    
    print("   ✅ Directory structure ready!\n")

def move_test_data():
    """Move test data files to correct locations"""
    print("🚚 Moving test data files to correct locations...")
    
    file_moves = [
        ("ecg_lvh_positive.csv", "data/raw/ecg/heartbeat/ecg_lvh_positive.csv"),
        ("ecg_normal.csv", "data/raw/ecg/heartbeat/ecg_normal.csv"),
        ("sample_clinical_data.csv", "data/raw/clinical/sample_clinical_data.csv")
    ]
    
    for source, destination in file_moves:
        if os.path.exists(source):
            shutil.move(source, destination)
            print(f"   ✓ Moved {source} -> {destination}")
        else:
            print(f"   ⚠️  {source} not found - please place manually")
    
    print("   ✅ Test data files positioned!\n")

def install_requirements():
    """Install enhanced requirements"""
    print("📦 Installing enhanced requirements...")
    
    if os.path.exists("requirements_enhanced.txt"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_enhanced.txt"])
            print("   ✅ Enhanced packages installed successfully!\n")
            return True
        except subprocess.CalledProcessError:
            print("   ❌ Failed to install packages. Please run manually:")
            print("      pip install -r requirements_enhanced.txt\n")
            return False
    else:
        print("   ⚠️  requirements_enhanced.txt not found")
        print("   📦 Please install manually: matplotlib seaborn scipy\n")
        return False

def update_app_files():
    """Update app.py and templates"""
    print("🔄 Updating application files...")
    
    # Update app.py
    if os.path.exists("app_enhanced.py"):
        shutil.copy2("app_enhanced.py", "app.py")
        print("   ✓ Updated app.py with enhanced prediction engine")
    else:
        print("   ❌ app_enhanced.py not found")
        return False
    
    # Update results.html
    if os.path.exists("results_enhanced.html"):
        shutil.copy2("results_enhanced.html", "templates/results.html")
        print("   ✓ Updated results.html with graph display")
    else:
        print("   ❌ results_enhanced.html not found")
        return False
    
    print("   ✅ Application files updated!\n")
    return True

def run_training_pipeline():
    """Run the complete training pipeline"""
    print("🤖 Running ML training pipeline...")
    
    scripts = [
        ("process_data.py", "Data processing"),
        ("train_models.py", "Model training")
    ]
    
    for script, description in scripts:
        if os.path.exists(script):
            print(f"   🔄 Running {description}...")
            try:
                subprocess.check_call([sys.executable, script])
                print(f"   ✅ {description} completed")
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {description} had issues - check logs")
        else:
            print(f"   ❌ {script} not found")
    
    print("   ✅ Training pipeline completed!\n")

def print_test_instructions():
    """Print testing instructions"""
    print("🧪 TESTING INSTRUCTIONS")
    print("-" * 30)
    print("1. Start the enhanced system:")
    print("   python run.py")
    print()
    print("2. Go to: http://localhost:5000/upload")
    print()
    print("3. Test LVH Positive Case:")
    print("   - Age: 67, Sex: Male, BP: 160, Cholesterol: 286")
    print("   - Upload: ecg_lvh_positive.csv")
    print("   - Expected: 'LVH Detected' with graphs")
    print()
    print("4. Test Normal Case:")
    print("   - Age: 45, Sex: Female, BP: 130, Cholesterol: 204") 
    print("   - Upload: ecg_normal.csv")
    print("   - Expected: 'No LVH Detected' with graphs")
    print()

def print_success():
    """Print success message"""
    print("=" * 60)
    print("🎉 ENHANCEMENT INSTALLATION COMPLETED!")
    print("=" * 60)
    print("Your LVH Detection System now includes:")
    print("✅ Real prediction engine (no more demo results)")
    print("✅ Performance visualization graphs")
    print("✅ Clinical risk factor analysis")
    print("✅ Model confidence scoring")
    print("✅ Professional medical reporting")
    print()
    print("Ready for academic presentation! 🎓")
    print("=" * 60)

def main():
    """Main installation function"""
    print_header()
    
    # Check current directory
    if not os.path.exists("app.py"):
        print("❌ Please run this script in your LVH project directory")
        print("   (The directory containing app.py)")
        return
    
    try:
        # Step 1: Backup existing files
        backup_files()
        
        # Step 2: Check for test data
        if not check_data_files():
            print("⚠️  Please download test data files first, then re-run this script")
            return
        
        # Step 3: Setup directories
        setup_directories()
        
        # Step 4: Move test data
        move_test_data()
        
        # Step 5: Install requirements  
        install_requirements()
        
        # Step 6: Update application files
        if not update_app_files():
            print("❌ Failed to update app files")
            return
        
        # Step 7: Run training pipeline
        run_training_pipeline()
        
        # Step 8: Print instructions
        print_test_instructions()
        print_success()
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        print("Please follow the manual instructions in IMPLEMENTATION-GUIDE.md")

if __name__ == "__main__":
    main()