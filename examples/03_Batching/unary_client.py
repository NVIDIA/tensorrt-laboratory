import grpc

import simple_pb2
import simple_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50049') as channel:
        stub = simple_pb2_grpc.InferenceStub(channel)
        
        response  = stub.Compute(simple_pb2.Input(batch_id=78))
        print("Received msg  with batch_id={}".format(response.batch_id))

if __name__ == "__main__":
    run() 
