# Datawriters

Here go scripts that use a ZMQ SUB socket to write data to a dataset.

The script can be any language, but it must accept two sys arguments, PORT and RUN_ID. Example: `$ python writer.py PORT RUN_ID`.

The script must connect a ZMQ SUB socket to PORT (on localhost) and a ZMQ REQ socket to PORT+1. The synchronization is then
  * Connect SUB
  * Send ping (empty message) on the REQ
  * Receive ping on the REQ
  * Start receiving on the SUB

The RUND_ID should in the future be GUID.