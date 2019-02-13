# Streaming Examples

Async gRPC streaming can take on many forms. nvRPC provides a set of LifeCycles
to accommodate a variety of common use-cases.

For all examples, the RPC that we will implement will have the same form:

```protobuf
rpc Bidirectional (stream Input) returns (stream Output) {}
```

To implement a `StreamingContext`, you must implement the `ReceivedRequest` pure
virtual method.  This method is triggered once for each incoming request that is
read from the stream.  The `ServerStream` object is used to write responses or 
close the stream.  

```c++
class SimpleContext final : public StreamingContext<Input, Output, ExternalResources>
{
    void RequestReceived(RequestType&& input, std::shared_ptr<ServerStream> stream) final override
    {
       // `input` incoming message of Input/RequestType
       // `stream` allows you to:
       //  ->StreamID()
       //    - unique identifier for this stream; note, the ID will be reused when the context is
       //      recycled.
       //  ->WriteResponse(Output&&)
       //    - writes a response on the stream; returns `true` if the stream is connected;
       //      otherwise, `false`  if the stream is disconnected
       //  ->IsConnected()
       //    - bool - is the stream still connected to the client. `FinishStream` or `CancelStream`
       //      will disconnect all `ServerStream`  objects that share the same StreamID.
       //  ->FinishStream()
       //    - Close the Stream from the server-side with status OK
       //  ->CancelStream()
       //    - Close the Stream with status CANCELLED
       //
       //  NOTE: The gRPC stream will stay open to the client until:
       //        1) the client closes its half of the stream, and
       //        2) all `ServerStream` objects are destroyed OR
       //           the stream is explicitly closed by Cancel/FinishStream
    }
```

The final comment is worth further discussion.  The life of the `ServerStream` object does
not have to be tied to the life of the `ReceivedRequest` call.  You can pass the
`ServerStream` object off to an external resource which can then write messages to on the
stream has long as the stream remains connected.  

The stream will disconnect implicitly 


- `ping-pong.cc` - In-order send/recv stream. This client sends a Request and
  the server Response with the same value for `batch_id`.  