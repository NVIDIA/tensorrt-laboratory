# Shared Memory Service

Client/Server service extending the Basic nvRPC example.

The client (`sysv-client.x`) creates a `CyclicAllocator<SystemV>` from which it
allocates buffers of shared memory.

The client:
  1. Write some data into shared memory (batch_id and 0xDEADBEEF), `data[0]` and `data[1]` respectively.
  2. Packs the RPC message with the batch_id and the SystemV memory descriptor details
  3. Initiates the RPC
  4. Checks the value of the address in which it wrote 0xDEADBEEF to ensure that the server wrote it's response to the proper location.

On reciept of a client RPC, the server Aquires a `Descriptor<SystemV>` to the
offset in the shared memory segment required in the RPC.  This is done via the
`ExternalSharedMemoryManager`.

The server:
  1. Aquires the `Descriptor<SystemV>` to the offset
  2. Checks the values at `data[0]` and `data[1]` for `batch_id` and `0xDEADBEEF` respectively
  3. Write the `batch_id` into `[1]` element
  4. Returns the Response