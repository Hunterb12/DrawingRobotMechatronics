import subprocess

def main():
    subprocess.run(["python","image_to_gcode.py","--input", "gatech.png","--output", "graph.nc", "--threshold", "100"])












if __name__ == '__main__':
    main()