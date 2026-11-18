"""Strict private UniProc command and receipt route; physical readiness gated."""
from .transport import require, identity


class WorkerRoute:
    def __init__(self, incoming, outgoing, backend):
        require(incoming.direction == 'engine-worker' and outgoing.direction == 'worker-engine',
                'private worker directions required')
        self.incoming, self.outgoing, self.backend = incoming, outgoing, backend

    def handle(self, frame):
        kind, body = self.incoming.receive(frame)
        self.backend.verify_identity()
        # An already submitted generation must remain cancellable after a
        # readiness failure. The backend verifies exact receipt/binding ownership.
        if body['operation'] != 'cancel':
            self.backend.require_live()
        # Never dispatch an arbitrary method name from the frame.
        operations = {'prepare': self.backend.prepare, 'append': self.backend.append,
                      'complete': self.backend.complete, 'cancel': self.backend.cancel}
        receipt = operations[body['operation']](body)
        self.backend.verify_receipt(body, receipt)
        result = {k: body[k] for k in ('request','session','turn','generation','operation','binding')}
        result['receipt'] = receipt
        return self.outgoing.send('worker_receipt', result)


class WorkerClient:
    def __init__(self, outgoing, incoming, executor, verify_receipt):
        self.outgoing, self.incoming = outgoing, incoming
        self.executor, self.verify_receipt = executor, verify_receipt

    def call(self, body):
        # UniProc executes in EngineCore's process; this adds no worker process.
        require(type(self.executor).__module__ == 'vllm.v1.executor.uniproc_executor'
                and type(self.executor).__name__ == 'UniProcExecutor', 'UniProc-only dispatch')
        values = self.executor.collective_rpc('remote_ra_private_v2_dispatch',
                    args=(self.outgoing.send('worker_command', body),), timeout=30)
        require(type(values) is list and len(values) == 1, 'one private worker result required')
        kind, result = self.incoming.receive(values[0])
        require(kind == 'worker_receipt' and identity(result) == identity(body)
                and result['binding'] == body['binding'] and result['operation'] == body['operation'],
                'worker receipt correlation mismatch')
        self.verify_receipt(body, result['receipt'])
        return result['receipt']
