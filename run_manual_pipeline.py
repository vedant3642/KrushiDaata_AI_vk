import os
import sys
import subprocess

def main():
    print("=" * 80)
    print("🌾 KRUSHIDAATA: MANUAL EXECUTION & TRAINING PIPELINE 🌾")
    print("=" * 80)
    print("Select an action to execute:")
    print("  [1] Train PyTorch Multi-Task DNN Model (train_torch.py)")
    print("  [2] Train TensorFlow / Keras Multi-Task Model (train_tensor.py)")
    print("  [3] Compare Saved Models (compare_three_models.py)")
    print("  [4] Run Regional Sanity & Area Share Benchmark (evaluate_regional_sanity.py)")
    print("  [5] Launch Interactive Streamlit Web Application (app.py)")
    print("  [0] Exit")
    print("=" * 80)
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip()
    else:
        try:
            choice = input("Enter choice [0-5]: ").strip()
        except EOFError:
            choice = "4"
            
    py_exec = sys.executable
    
    if choice == "1":
        print("\n🚀 Running PyTorch Model Training...")
        subprocess.run([py_exec, "train_torch.py"], check=True)
    elif choice == "2":
        print("\n🚀 Running TensorFlow Keras Model Training...")
        subprocess.run([py_exec, "train_tensor.py"], check=True)
    elif choice == "3":
        print("\n🚀 Running Model Comparison Benchmark...")
        subprocess.run([py_exec, "compare_three_models.py"], check=True)
    elif choice == "4":
        print("\n🚀 Running Regional Sanity Evaluation Benchmark...")
        subprocess.run([py_exec, "evaluate_regional_sanity.py"], check=True)
    elif choice == "5":
        print("\n🚀 Launching Streamlit Web App...")
        subprocess.run(["streamlit", "run", "app.py"])
    elif choice == "0":
        print("Exiting pipeline manager.")
    else:
        print("Invalid option selected.")

if __name__ == "__main__":
    main()
