import zmq


class ConnectionBuilder():
    def __init__(self):
        self.NFC_Id_port=9000
        self.NFC_PUB_req=8000
        self.NFC_SUB_customer=6000

    def get_sub_12d(self):
        ctx = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        #sock.connect('tcp://127.0.0.1:9000')
        sock.connect('tcp://127.0.0.1:'+str(self.NFC_Id_port))
        sock.subscribe("")
        #sock.setsockopt_string(zmq.SUBSCRIBE,'');print("subscribe port is created to receive NFC_id")

        print("receiving")
        #message=sock.recv_pyobj()
        #print("reci")
        #NFC_ID=message.get('NFC_ID')
        NFC_ID = sock.recv_string()
        #print("Received string: %s ..." % NFC_ID )
        return (NFC_ID)

    def pub_NFC_id(self,NFC_ID):
        context_NFC=zmq.Context()
        socket_NFC_out=context_NFC.socket(zmq.PUB)
        socket_NFC_out.bind('tcp://127.0.0.1:'+str(self.NFC_PUB_req))

        socket_NFC_out.send_string(NFC_ID)
        print("Sent string:{0}...".format(NFC_ID))

    def get_customer_details(self):
        context_NFC_res=zmq.Context()
        socket__NFC_res=context_NFC_res.socket(zmq.SUB)
        socket__NFC_res.connect('tcp://127.0.0.1:'+str(self.NFC_SUB_customer))
        socket__NFC_res.setsockopt_string(zmq.SUBSCRIBE,'');print("subscribe port is created to receive customer_details")

        CustomerDetiles = socket__NFC_res.recv_string()
        #print("Received string: %s ..." % NFC_ID )
        return (CustomerDetiles)

def main_NFC():
    print("Receiving 12 digit NFC-ID")
    connection = ConnectionBuilder()
    NFC_id = connection.get_sub_12d()
    print("Received 12 digit NFC-ID: %s " % NFC_id)
    connection.pub_NFC_id(NFC_id)
    print("Publish_NFC_id to request customer details")
    Customer_Detiles = connection.get_customer_details()
    print("Recevied Customer Detiles")
    print(Customer_Detiles)


if __name__ == '__main__':
    main_NFC()
