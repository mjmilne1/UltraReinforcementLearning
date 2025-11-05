"""Launch the Ultra RL Trading Dashboard"""
import subprocess
import sys

print("="*60)
print("🚀 Launching Ultra RL Trading Dashboard")
print("="*60)
print("\nThe dashboard will open in your browser automatically.")
print("If not, navigate to: http://localhost:8501\n")
print("Press Ctrl+C to stop the dashboard\n")

# Run streamlit
subprocess.run([sys.executable, "-m", "streamlit", "run", "src/dashboard/dashboard.py"])
