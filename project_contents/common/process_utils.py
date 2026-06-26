import psutil
import subprocess
import socket
from psutil import process_iter
from signal import SIGTERM # or SIGKILL

def run_seperate_command(command):
    subprocess.Popen(command, shell=True)
def free_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    proc.terminate()  # Terminate the process
                    print(f"Process {proc.info['name']} (PID: {proc.info['pid']}) terminated.")
                    return
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    print(f"No process found running on port {port}.")

# def free_port(port):
#     try:
#         # Create a socket object
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             # Set socket options to allow reuse of the port
#             s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#             # Bind the socket to the port
#             s.bind(('localhost', port))
#             print(f"Port {port} is now free.")
#     except OSError as e:
#         if e.errno == 98:
#             print(f"Port {port} is still in use.")
#         else:
#             print(f"An error occurred: {e}")