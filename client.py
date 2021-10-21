import socket
import json


sk = socket.socket(type=socket.SOCK_DGRAM)  # udp协议
sk.sendto('hello'.encode('utf-8'), ('127.0.0.1', 9000))

# msg_recv, addr = sk.recvfrom(1024)
# print(msg_recv.decode('utf-8'))
msg_recv = sk.recv(1024)
print(msg_recv.decode('utf-8'))

sk.close()

