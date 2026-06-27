import socket
import sys

def main():
    print("=======================================")
    print("      PYTHONCRAFT DEBUG CONSOLE")
    print("=======================================")
    print("Kullanim:")
    print("  tp <x> <z> <y>   - Oyuncuyu isinlar (Y yuksekliktir)")
    print("  exit             - Konsolu kapatir")
    print("Ornek: tp 1000 1000 120 (X=1000, Z=1000, Y=120 yukseklik)")
    print("=======================================\n")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('127.0.0.1', 25565)
    
    while True:
        try:
            cmd = input(">> ")
            if cmd.lower().strip() == 'exit':
                break
            if cmd.strip():
                sock.sendto(cmd.encode('utf-8'), server_address)
        except (KeyboardInterrupt, EOFError):
            break
            
if __name__ == "__main__":
    main()
