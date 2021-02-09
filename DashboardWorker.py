import paho.mqtt.client as mqtt
import time
import sys

dev = sys.argv[1]
sani = sys.argv[2]
crowd = sys.argv[3]
seq = sys.argv[4]
socket = sys.argv[5]
Clap = sys.argv[6]

if(dev == "x"):
    nano_san = "VToPdTKsbLByBC9Xf1vn"
    nano_crowd="x8ia7aaWK131CviB4a3B"    
    nano_seq = "72nbbQFD8wPvKZ8VOGvl"
    ec2_socket = "MtullRL69vnCPkOpqcba"
elif(dev == "u"):
    nano_san = "HJYNkW3f8MoAjKYReQKk"
    #nano_san =  "VToPdTKsbLByBC9Xf1vn"
    #nano_crowd= "x8ia7aaWK131CviB4a3B"
    nano_seq =  "PaJR1Koq7lnlJHY03e8A"
    ec2_socket = "MtullRL69vnCPkOpqcba"


#VToPdTKsbLByBC9Xf1vn
#x8ia7aaWK131CviB4a3B
#HJYNkW3f8MoAjKYReQKk
#PaJR1Koq7lnlJHY03e8A

client = mqtt.Client()
client.username_pw_set(nano_san)
client.connect('ec2-3-231-140-244.compute-1.amazonaws.com', 1883, 1)
text_sanit = "\"sanit\":"
text_format = "{" + text_sanit + '"' + sani + '"' + "}"
ret = client.publish('v1/devices/me/telemetry', text_format)  # change the value of TESTING string
print(str(ret), text_format)


client.username_pw_set(nano_san)
client.connect('ec2-3-231-140-244.compute-1.amazonaws.com', 1883, 1)
text_crowd = "\"crowd\":"
text_format = "{" + text_crowd + '"' + crowd + '"' + "}"
ret = client.publish('v1/devices/me/telemetry', text_format)  # change the value of TESTING string
print(str(ret), text_format)


client.username_pw_set(nano_seq)
client.connect('ec2-3-231-140-244.compute-1.amazonaws.com', 1883, 1)
text_seq = "\"val1\":"
text_format = "{" + text_seq + '"' + seq + '"' + "}"
ret = client.publish('v1/devices/me/telemetry', text_format)  # change the value of TESTING string
print(str(ret), text_format)


client.username_pw_set(ec2_socket)
client.connect('ec2-3-231-140-244.compute-1.amazonaws.com', 1883, 1)
text_socket = "\"valx\":"
text_format = "{" + text_socket + socket + "|"+ Clap + "}"
ret = client.publish('v1/devices/me/telemetry', text_format)
print(str(ret), text_format) # change the value of TESTING string"""

