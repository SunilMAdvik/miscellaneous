#process-1 NFC-Reader -NFc-ID= 12 digit code
# <server> - custore detials - publish

import zmq

    def main_NFC():
        print("Receiving 12 digit NFC-ID")
        NFC_id = sub_12d()
        print("Received 12 digit NFC-ID: %s ..." % NFC_id)


        (out_port_NFC, socket_NFC_out), (in_port_NFC, socket__NFC_res) = ConnectionBuilder.get_connections_to_NFC()
        socket_NFC_out.send_pyobj({'image':window})
        message = socket__NFC_res.recv_pyobj()
        CustomerDetiles = message.get( ? )

class ConnectionBuilder():
    def __init__(self):
        self.NFC_Id_port=9000
        self.NFC_pub_port=6000
        self.NFC_sub_port=8000


    def get_connections_to_NFC(self):
        context_NFC=zmq.Context()
        socket_NFC_out=context_NFC.socket(zmq.PUB)
        socket_NFC_out.bind('tcp://127.0.0.1:'+str(self.NFC_pub_port))

        context_NFC_res=zmq.Context()
        socket__NFC_res=context_NFC_res.socket(zmq.SUB)
        socket__NFC_res.connect('tcp://127.0.0.1:'+str(self.NFC_sub_port))
        socket__NFC_res.setsockopt_string(zmq.SUBSCRIBE,'')

        return (self.NFC_pub_port, socket_NFC_out), (self.NFC_sub_port,socket__NFC_res)

    def sub_12d():
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect('tcp://127.0.0.1:9000')
        sock.subscribe("")

        NFC_ID = sock.recv_string()
        #print("Received string: %s ..." % NFC_ID )
        return (NFC_ID)



if __name__ == '__main__':
