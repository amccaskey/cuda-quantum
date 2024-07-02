import cudaq
import cupy as cp

cudaq.set_target('tensornet-mps')


@cudaq.kernel
def test(n: int):
    q = cudaq.qvector(n)
    h(q[0])


state = cudaq.get_state(test, 10)
tensors = state.getTensors()
arrays = cudaq.to_cupy(state, dtype=cp.complex128)
for a in arrays:
    print(a)

for t in tensors:
    print(t.get_rank(), t.extents)

# # For N ranks, we'll take an M qubit state and decompose across
# # the N ranks (N QPUs). Each QPU will track their portion of the
# # MPS tensors, and track additional tensors

# QPU 0 <--> QPU 1 <--> QPU 2 <--> QPU 3 <--> ... <--> QPU N-1

class QPU(object):
    rank: int
    tensors = []
    num_local_qubits = 0
    num_global_qubits = 0

    def __init__(self, rank, numGlobalQubits, numLocalQubits, numQPUs):
        self.rank = rank
        self.num_local_qubits = numLocalQubits
        self.num_global_qubits = numGlobalQubits

        self.qubit_map = {
            globalId: k
            for k, globalId in enumerate(
                range(rank * (numLocalQubits - 1),
                      rank * (numLocalQubits - 1) + numLocalQubits - 1))
        }
        print(self.rank, self.qubit_map)

        self.tensors = [cp.zeros((2, 1), dtype=cp.complex128)]
        self.tensors[-1][0, 0] = 1.
        if self.rank != 0:
            # Sync tensor
            self.tensors.append(cp.zeros((1, 2, 1), dtype=cp.complex128))
            self.tensors[-1][0, 0, 0] = 1.

        for _ in range(self.num_local_qubits):
            qubitTI = cp.zeros((1, 2, 1), dtype=cp.complex128)
            qubitTI[0, 0, 0] = 1.
            self.tensors.append(qubitTI)

        # Sync tensor
        if self.rank != numQPUs - 1:
            self.tensors.append(cp.zeros((1, 2, 1), dtype=cp.complex128))
            self.tensors[-1][0, 0, 0] = 1.

        self.tensors.append(cp.zeros((1, 2), dtype=cp.complex128))
        self.tensors[-1][0, 0] = 1.

    def get_local_qubit_id(self, globalQubit: int):
        # each qpu has numGlobal / numQPUs qubits
        return self.qubit_map[globalQubit]

    def is_boundary_qubit(self, globalQubit: int) -> bool:
        local = self.get_local_qubit_id(globalQubit)
        return local == 0 or local == self.num_local_qubits - 1

    def get_rank(self):
        return self.rank

    def applySingleQubitOperation(self, globalQubitId, operation):
        if not globalQubitId in self.qubit_map: return
        localQubit = self.get_local_qubit_id(globalQubitId)
        if localQubit > self.num_local_qubits - 1:
            raise RuntimeError(
                f"invalid qubit index ({localQubit}) on {self.rank}")
        numSyncTensors = len(self.tensors) - 2 - self.num_local_qubits
        tranformedQubitIdx = localQubit + numSyncTensors

        k, state = cudaq.make_kernel(cudaq.State)
        q = k.qalloc(state)
        getattr(k, operation)(q[tranformedQubitIdx])

        print(self.tensors)
        fromDataState = cudaq.State.from_data(self.tensors)
        self.currentState = cudaq.get_state(k, fromDataState)
        # Are the tensors changed under the hood?
        newTensors = cudaq.to_cupy(self.currentState, dtype=cp.complex128)
        self.tensors = newTensors
        print(self.tensors)

    def h(self, globalQubitId):
        self.applySingleQubitOperation(globalQubitId, 'h')


# Number of global qubits
N = 8
# Number of local qubits
M = 4
# Number of QPUs
Q = 4

qpus = [QPU(i, N, M, Q) for i in range(Q)]

# Gates operate on global qubits
[qpus[0].h(i) for i in range(4)]
