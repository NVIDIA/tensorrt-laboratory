import grpc

import simple_pb2
import simple_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = simple_pb2_grpc.InferenceStub(channel)
        def requests():
            messages = [simple_pb2.Input(batch_id=i) for i in range(10)]
            for msg in messages:
                print("Sending Stream batch_id={}".format(msg.batch_id))
                yield msg

        responses = stub.BatchedCompute(requests())
        for resp in responses:
            print("Received msg on stream with batch_id={}".format(resp.batch_id))

if __name__ == "__main__":
    run() 
