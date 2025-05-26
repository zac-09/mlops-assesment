#!/usr/bin/env python3
"""
Alternative deployment script using Cerebrium API directly.
"""

import subprocess
import sys
import os
import json

def deploy_with_cli():
    """Deploy using cerebrium CLI with simplified approach."""
    
    print("🚀 Deploying to Cerebrium...")
    
    # Try deploying with minimal config
    try:
        # First, let's see what cerebrium deploy commands are available
        result = subprocess.run(['cerebrium', 'deploy', '--help'], 
                              capture_output=True, text=True)
        print("Available deploy options:")
        print(result.stdout)
        
        # Try simple deploy command
        print("\nAttempting deployment...")
        result = subprocess.run([
            'cerebrium', 'deploy',
            '--name', 'imagenet-classifier',
            '--python-version', '3.9',
            '--gpu', 'T4',
            '--dockerfile', 'Dockerfile'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Deployment successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Deployment failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error during deployment: {e}")
        return False

def deploy_without_config():
    """Try deploying without config file."""
    
    # Rename config file temporarily
    config_file = "cerebrium.toml"
    backup_file = "cerebrium.toml.backup"
    
    if os.path.exists(config_file):
        os.rename(config_file, backup_file)
    
    try:
        print("🚀 Trying deployment without config file...")
        result = subprocess.run(['cerebrium', 'deploy'], 
                              capture_output=True, text=True, input="y\n")
        
        if result.returncode == 0:
            print("✅ Deployment successful!")
            print(result.stdout)
            return True
        else:
            print("❌ Deployment failed:")
            print(result.stderr)
            return False
            
    finally:
        # Restore config file
        if os.path.exists(backup_file):
            os.rename(backup_file, config_file)

def main():
    """Main deployment function."""
    
    print("Cerebrium Deployment Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    required_files = ['main.py', 'model.py', 'Dockerfile', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Try different deployment methods
    deployment_methods = [
        ("CLI with parameters", deploy_with_cli),
        ("Without config file", deploy_without_config),
    ]
    
    for method_name, method_func in deployment_methods:
        print(f"\n🔄 Trying: {method_name}")
        if method_func():
            print(f"✅ Success with: {method_name}")
            return True
        print(f"❌ Failed with: {method_name}")
    
    print("\n❌ All deployment methods failed.")
    print("💡 Try manual deployment:")
    print("   1. cerebrium deploy")
    print("   2. Follow the prompts")
    print("   3. Choose Dockerfile deployment")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)