# Stein-Variational-Gradient-Descent

Survey of Stein Variational Gradient Descent 

Files contained in this repository:

1. Jupyter Notebook assignment on SVGD. If you would like access to the memo, please contact me at leon.halgryn@gmail.com
2. Background folder containing 2 Jupyter notebooks:
   a. Background information on measure theory for SVGD
   b. Background information on kernels and reproducing kernel Hilbert spaces for SVGD
3. Report on SVGD.   
4. Experiments folder containing code to reproduce the results given in report. Contains 2 subfolders:
   a. SVGD. This folder contains the code to reproduce SVGD results given in report. To reproduce the results, navigate to the local folder and run:
     - 'python sampling.py' and
     - 'python variance_collapse.py'
   b. SVPG. This folder contains the code to reproduce SVPG results given in report. To reproduce the results, navigate to the local folder and run:
      - 'python acrobot.py' and
      - 'python cartpole.py' and
      - 'python lunar_lander.py'

Note that the lunar_lander.py file may give issues due to uninstalled dependencies. If such errors arise, the google colab notebook contains all the SVPG code to reproduce the results given in the report.
