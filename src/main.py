import os
import subprocess


def install_requirements():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


if __name__ == "__main__":
    install_requirements()
    print("Requirements installed successfully.")