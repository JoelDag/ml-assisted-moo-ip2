import os
import subprocess
from src.IP2.evolutionaryComputation import evolutionaryRunner
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def install_requirements():
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


if __name__ == "__main__":
    install_requirements()
    runner = evolutionaryRunner(pop_size=100, n_gen=200,n_var=2, m_obj=2, t_past=10,
         t_freq=5, test_problem="makeMMF1Function", jutting_param=1.1, h_interval=3)
    runner.run()