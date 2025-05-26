#!/usr/bin/env python3
"""
Simple script to download model weights for Docker build.
"""

import os
import urllib.request

def main():
    weights_path = "pytorch_model_weights.pth"
    
    if not os.path.exists(weights_path):
        print("Downloading model weights...")
        url = "https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1"
        
        try:
            urllib.request.urlretrieve(url, weights_path)
            
            # Verify file size
            if os.path.getsize(weights_path) > 1000000:  # Should be > 1MB
                print("Weights downloaded successfully!")
            else:
                print("Downloaded file seems too small")
                os.remove(weights_path)
                raise Exception("Invalid download")
                
        except Exception as e:
            print(f"Failed to download weights: {e}")
            raise
    else:
        print("Weights already present")

if __name__ == "__main__":
    main()