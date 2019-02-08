# BidirectionalStream In-Order Send/Recv

```
rpc InOrderSendRecv (stream Request) returns (stream Response)
```

The service will accept a stream or Requests, queue them for in-order execution
via the `ExecuteRPC` virtual method, and for each result of `ExecuteRPC` return
a Response on the stream.

Only one `ExecuteRPC` can be in-action at anytime which allows the RPC to
optionally maintain a state.  In some regards, this lifecyle could be used to
model an RNN as the collective history can be built into the resources.

