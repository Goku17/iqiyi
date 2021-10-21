import socket
import json


sk = socket.socket(type=socket.SOCK_DGRAM)  # udp协议
sk.bind(('127.0.0.1', 9000))

msg_recv, addr = sk.recvfrom(1024)
print(msg_recv.decode('utf-8'))
sk.sendto('收到'.encode('utf-8'), addr)

sk.close()
