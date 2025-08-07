
#!/usr/bin/env python3
"""
Quick script to save the currently trained model from the backtesting system.
Run this after a successful backtest to save the model.
"""

import requests
import json
import sys

def save_current_model(custom_name=None):
    """Save the current trained model"""
    try:
        url = "http://localhost:5000/save_model"
        data = {}
        
        if custom_name:
            data['filename'] = custom_name
        
        response = requests.post(url, json=data, timeout=10)
        result = response.json()
        
        if result.get('status') == 'success':
            print(f"✅ {result['message']}")
            print(f"📁 Saved to: {result['filepath']}")
            return True
        else:
            print(f"❌ Error: {result.get('message', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to the trading bot server.")
        print("   Make sure the bot is running on port 5000.")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def list_saved_models():
    """List all saved models"""
    try:
        url = "http://localhost:5000/list_models"
        response = requests.get(url, timeout=10)
        result = response.json()
        
        models = result.get('models', [])
        if not models:
            print("📂 No saved models found.")
            return
        
        print(f"📂 Found {len(models)} saved models:")
        print("-" * 80)
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['filename']}")
            print(f"   Type: {model['model_type']}")
            print(f"   Accuracy: {model['training_accuracy']:.4f}")
            print(f"   Saved: {model['saved_at']}")
            print(f"   Features: {len(model['features'])}")
            print()
            
    except Exception as e:
        print(f"❌ Error listing models: {str(e)}")

if __name__ == "__main__":
    print("🤖 Bitcoin Trading Model Saver")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_saved_models()
            sys.exit(0)
        else:
            custom_name = sys.argv[1]
    else:
        custom_name = None
    
    print("💾 Attempting to save current trained model...")
    
    if save_current_model(custom_name):
        print("\n📋 To list all saved models, run:")
        print("python save_current_model.py list")
    else:
        print("\n💡 Tips:")
        print("1. Make sure the trading bot is running")
        print("2. Run a backtest first to train a model")
        print("3. Then run this script to save it")
