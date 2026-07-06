import os
import subprocess
import sys
import shutil

def build_app():
    print("==================================================")
    print("      PERABEATS Photo Selecter Builder            ")
    print("==================================================")
    
    # 1. Check if PyInstaller is installed
    try:
        import PyInstaller
        print(f"[OK] PyInstaller is installed (v{PyInstaller.__version__})")
    except ImportError:
        print("[FAIL] PyInstaller is not installed in this environment.")
        print("Please run: pip install pyinstaller")
        return

    # 2. Check if the source file exists
    source_file = "desktop_app.py"
    if not os.path.exists(source_file):
        print(f"[FAIL] Source file '{source_file}' not found.")
        return
        
    # 3. Base command configuration
    # We use --onedir (directory mode) because PyTorch is huge (~1GB+).
    # Single-file mode (--onefile) takes 30-40 seconds to start up on every run 
    # since it has to decompress 1GB of DLLs to a temp directory first.
    # Directory mode starts instantly and can be easily shared as a ZIP folder!
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name=PERABEATS_Photo_Selecter",
        "--collect-all=torch",
        "--collect-all=torchvision",
        "--collect-all=rawpy",
        "--collect-all=PIL",
        "--collect-all=pyiqa"
    ]
    
    # 4. Check and include model weights
    weights_file = "epoch-82.pth"
    if os.path.exists(weights_file):
        print(f"[OK] Found local weights '{weights_file}'. Bundling it with the app...")
        cmd.append(f"--add-data={weights_file};.")
    else:
        print(f"[WARN] Warning: '{weights_file}' not found. The app will fallback to PyIQA online model hub if needed.")

    # 5. Check and include app logo
    logo_file = "perabeats_logo.png"
    if os.path.exists(logo_file):
        print(f"[OK] Found app logo '{logo_file}'. Bundling it with the app...")
        cmd.append(f"--add-data={logo_file};.")
    else:
        print(f"[WARN] Warning: Logo file '{logo_file}' not found.")

    cmd.append(source_file)
    
    print("\nRunning PyInstaller compilation (this may take a few minutes as PyTorch is large)...")
    print("Command:", " ".join(cmd))
    
    try:
        # Run PyInstaller
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Read and print output in real-time
        for line in process.stdout:
            print(line, end="")
            
        process.wait()
        
        if process.returncode == 0:
            print("\n==================================================")
            print("         BUILD SUCCESSFUL! [COMPLETED]            ")
            print("==================================================")
            dist_path = os.path.abspath(os.path.join("dist", "PERABEATS_Photo_Selecter"))
            print(f"\nPortable app folder created at:\n{dist_path}\n")
            print("To share it with friends:")
            print("1. Open the 'dist' folder.")
            print("2. Right-click the 'PERABEATS_Photo_Selecter' folder -> Send to -> Compressed (zipped) folder.")
            print("3. Send the ZIP file to your friends.")
            print("4. They can extract the ZIP and double-click 'PERABEATS_Photo_Selecter.exe' to run offline!")
        else:
            print(f"\n[FAIL] Build failed with exit code: {process.returncode}")
            
    except Exception as e:
        print(f"\n[FAIL] An error occurred during the build: {e}")

if __name__ == "__main__":
    build_app()
