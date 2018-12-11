import subprocess

proc = subprocess.Popen(['python', 'Simulation.py',  '1500', '1000'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(proc.communicate()[0])