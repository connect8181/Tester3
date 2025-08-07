
import subprocess
import sys
import os
import signal
import time

def kill_existing_processes():
    """Kill any existing Flask processes"""
    try:
        subprocess.run(['pkill', '-f', 'python.*main.py'], check=False)
        time.sleep(2)  # Wait for processes to die
        print("✅ Killed existing processes")
    except Exception as e:
        print(f"⚠️ Error killing processes: {e}")

def start_bot():
    """Start the trading bot"""
    try:
        kill_existing_processes()
        print("🚀 Starting Bitcoin Trading Bot...")
        
        # Start the main application
        process = subprocess.Popen([sys.executable, 'main.py'])
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping bot...")
            process.terminate()
            process.wait()
            
    except Exception as e:
        print(f"❌ Error starting bot: {e}")

if __name__ == "__main__":
    start_bot()
