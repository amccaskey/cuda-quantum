import cudaq
import cupy as cp
from mpi4py import MPI

cudaq.set_target('tensornet-mps')

# @cudaq.kernel
# def test(n: int):
#     q = cudaq.qvector(n)
#     h(q[0])

# state = cudaq.get_state(test, 10)
# tensors = state.getTensors()
# arrays = cudaq.to_cupy(state, dtype=cp.complex128)
# for a in arrays:
#     print(a)

# for t in tensors:
#     print(t.get_rank(), t.extents)

# # For N ranks, we'll take an M qubit state and decompose across
# # the N ranks (N QPUs). Each QPU will track their portion of the
# # MPS tensors, and track additional tensors

# QPU 0 <--> QPU 1 <--> QPU 2 <--> QPU 3 <--> ... <--> QPU N-1


class QPU(object):
    rank: int
    tensors = []
    num_local_qubits = 0
    num_global_qubits = 0
    comm = MPI.COMM_WORLD

    def __init__(self, rank, qubit_map, rank_map):
        self.rank = rank
        self.rank_map = rank_map

        self.qubit_map = qubit_map
        self.num_local_qubits = len(qubit_map)

        # Infer the tracked global qubit ids (those that we
        # keep synced up to locally)
        globalQubits = [k for k, _ in self.qubit_map.items()]
        minGlobalQubit = min(globalQubits)
        maxGlobalQubit = max(globalQubits)
        self.trackedQubits = []
        if minGlobalQubit != 0:
            self.trackedQubits.append(minGlobalQubit - 1)
        if rank != self.comm.size - 1:
            self.trackedQubits.append(maxGlobalQubit + 1)

        print('RANK ', self.rank, self.trackedQubits)
        # Initialize this QPU's tensors. Should have 2 Dummy Tensors (rank 2)
        # NumLocalQubit Qubit State tensors (rank 3)
        # and 2 Sync Tensors (rank 3)

        # First Dummy tensor
        self.tensors = [cp.zeros((2, 1), dtype=cp.complex128)]
        self.tensors[-1][0, 0] = 1.

        # First Sync tensor
        self.tensors.append(cp.zeros((1, 2, 1), dtype=cp.complex128))
        self.tensors[-1][0, 0, 0] = 1.

        # NQubit state tensors
        for _ in range(self.num_local_qubits):
            qubitTI = cp.zeros((1, 2, 1), dtype=cp.complex128)
            qubitTI[0, 0, 0] = 1.
            self.tensors.append(qubitTI)

        # Sync tensor
        self.tensors.append(cp.zeros((1, 2, 1), dtype=cp.complex128))
        self.tensors[-1][0, 0, 0] = 1.

        # Final Dummy Tensor
        self.tensors.append(cp.zeros((1, 2), dtype=cp.complex128))
        self.tensors[-1][0, 0] = 1.

        # Now the "local state" is on NumLocal + 4 qubits

    def get_local_qubit_id(self, globalQubit: int):
        # each qpu has numGlobal / numQPUs qubits
        return self.qubit_map[globalQubit]

    def is_boundary_qubit(self, globalQubit: int) -> bool:
        local = self.get_local_qubit_id(globalQubit)
        return local == 0 or local == self.num_local_qubits - 1

    def get_rank(self):
        return self.rank

    def evolveState(self,
                    operation: str,
                    kernelQubitIndices: list,
                    parameters=None):
        # Create the Kernel, it takes an state to initialize
        # the allocated qvector to and applies the single qubit operation
        k, state = cudaq.make_kernel(cudaq.State)
        q = k.qalloc(state)
        args = [q[i] for i in kernelQubitIndices]
        if parameters != None:
            args = parameters + args

        getattr(k, operation)(*args)

        print(f'Tensors Before {self.rank}:\n{self.tensors}')

        # Create the state from our existing tensors
        fromDataState = cudaq.State.from_data(self.tensors)

        # Run the simulation, extract the new state. This should remain
        # on the GPu
        self.currentState = cudaq.get_state(k, fromDataState)

        # Get the state tensors as CuPy arrays
        newTensors = cudaq.to_cupy(self.currentState, dtype=cp.complex128)

        # Update the QPU tensor state
        self.tensors = newTensors

    def applySingleQubitOperation(self, operation,
                                  globalQubitId,
                                  parameters=None):
        # If we don't control this qubit, either return or wait for data
        if not globalQubitId in self.qubit_map:
            if globalQubitId in self.trackedQubits:
                # recieve the data
                data = cp.empty((1, 2, 1), dtype=cp.complex128)
                otherRank = self.rank_map[globalQubitId]
                # Actual tensor to recv is the qubit boundary from otherRank,
                # locally we have, and we want to sync either 1 or -2
                # [dummy]-[syncX-1]-[QBIT0]-...-[QBITN]-[syncX+1]-[dummy]
                idx = 1 if globalQubitId == self.trackedQubits[0] else -2
                print(
                    f'recv data from {otherRank} to {self.rank} with sync idx {idx} {data}'
                )
                self.comm.Recv(data, source=otherRank)
                print(f'success - {data}')
                self.tensors[idx] = data
            return

        # Get the local qubit id
        localQubit = self.get_local_qubit_id(globalQubitId)
        # The actual qubit index is offset by 2 since we have a
        # dummy tensor and a sync tensor
        self.evolveState(operation, [localQubit + 2], parameters=parameters)
        if self.is_boundary_qubit(globalQubitId) and globalQubitId > 0:
            # synchronize
            # Need the rank to send to... otherRank
            otherRank = self.rank - 1 if localQubit == 0 else self.rank + 1
            # Actual tensor to send is the qubit boundary,
            # [dummy]-[syncX-1]-[QBIT0]-...-[QBITN]-[syncX+1]-[dummy]
            # so either idx 2 or -3
            syncIdx = 2 if otherRank < self.rank else -3
            print(
                f'send {syncIdx} data from {self.rank} to {otherRank} - {self.tensors[syncIdx]}'
            )
            self.comm.Send(self.tensors[syncIdx], dest=otherRank)
        print(f'Tensors After {self.rank}:\n{self.tensors}')

    def applyTwoQubitOperation(self, operation: str, globalQubitId0: int,
                               globalQubitId1: int):
        # Could be local, i.e. globalQubitId 0 and 1 is here locally, we'll need to
        # sync only if they are boundary qubits and others should be tracking
        if self.rank_map[globalQubitId0] == self.rank_map[globalQubitId1]:

            # We have a local operation, apply and see if we need to
            # send/recieve boundaries
            if globalQubitId0 not in self.qubit_map:
                if globalQubitId0 in self.trackedQubits:
                    # recieve the data
                    data = cp.empty((1, 2, 1), dtype=cp.complex128)
                    otherRank = self.rank_map[globalQubitId0]
                    # Actual tensor to recv is the qubit boundary from otherRank,
                    # locally we have, and we want to sync either 1 or -2
                    # [dummy]-[syncX-1]-[QBIT0]-...-[QBITN]-[syncX+1]-[dummy]
                    idx = 1 if globalQubitId0 == self.trackedQubits[0] else -2
                    print(
                        f'recv data from {otherRank} to {self.rank} with sync idx {idx} {data}'
                    )
                    self.comm.Recv(data, source=otherRank)
                    print(f'success - {data}')
                    self.tensors[idx] = data
                    return

            if globalQubitId1 not in self.qubit_map:
                if globalQubitId1 in self.trackedQubits:
                    # recieve the data
                    data = cp.empty((1, 2, 1), dtype=cp.complex128)
                    otherRank = self.rank_map[globalQubitId1]
                    # Actual tensor to recv is the qubit boundary from otherRank,
                    # locally we have, and we want to sync either 1 or -2
                    # [dummy]-[syncX-1]-[QBIT0]-...-[QBITN]-[syncX+1]-[dummy]
                    idx = 1 if globalQubitId1 == self.trackedQubits[0] else -2
                    print(
                        f'recv data from {otherRank} to {self.rank} with sync idx {idx} {data}'
                    )
                    self.comm.Recv(data, source=otherRank)
                    print(f'success - {data}')
                    self.tensors[idx] = data
                    return
                
            if self.rank == self.rank_map[globalQubitId0]:
                localQubit0 = self.get_local_qubit_id(globalQubitId0)
                localQubit1 = self.get_local_qubit_id(globalQubitId1)
                self.evolveState(operation, [localQubit0 + 2, localQubit1 + 2])

                if self.is_boundary_qubit(
                        globalQubitId0) and globalQubitId0 > 0:
                    # synchronize
                    # Need the rank to send to... otherRank
                    otherRank = self.rank - 1 if localQubit0 == 0 else self.rank + 1
                    # Actual tensor to send is the qubit boundary,
                    # [dummy]-[syncX-1]-[QBIT0]-...-[QBITN]-[syncX+1]-[dummy]
                    # so either idx 2 or -3
                    syncIdx = 2 if otherRank < self.rank else -3
                    print(
                        f'send {syncIdx} data from {self.rank} to {otherRank} - {self.tensors[syncIdx]}'
                    )
                    self.comm.Send(self.tensors[syncIdx], dest=otherRank)
                    return

                if self.is_boundary_qubit(
                        globalQubitId1) and globalQubitId1 > 0:
                    # synchronize
                    # Need the rank to send to... otherRank
                    otherRank = self.rank - 1 if localQubit1 == 0 else self.rank + 1
                    # Actual tensor to send is the qubit boundary,
                    # [dummy]-[syncX-1]-[QBIT0]-...-[QBITN]-[syncX+1]-[dummy]
                    # so either idx 2 or -3
                    syncIdx = 2 if otherRank < self.rank else -3
                    print(
                        f'send {syncIdx} data from {self.rank} to {otherRank} - {self.tensors[syncIdx]}'
                    )
                    self.comm.Send(self.tensors[syncIdx], dest=otherRank)

            return

        # We are here - this is a remote operation.
        # Strategy will be to apply the operation on each QPU simultaneously
        localQubit = None
        syncTensorIdx = None
        qbits = []
        if globalQubitId0 in self.qubit_map:
            localQubit = self.get_local_qubit_id(globalQubitId0)
            otherRank = self.rank_map[globalQubitId1]
            syncTensorIdx = 1 if otherRank < self.rank else len(self.tensors)-2
            qbits += [localQubit+2, syncTensorIdx]
        elif globalQubitId1 in self.qubit_map:
            localQubit = self.get_local_qubit_id(globalQubitId1)
            otherRank = self.rank_map[globalQubitId0]
            syncTensorIdx = 1 if otherRank < self.rank else len(self.tensors)-2
            qbits += [syncTensorIdx, localQubit+2]
        else:
            # We are a QPU that has nothing to do with this
            return
        
        print('HI ', self.rank, qbits)

        self.evolveState(operation, qbits)
        return

    def h(self, globalQubitId):
        self.applySingleQubitOperation('h', globalQubitId)

    def cnot(self, q, r):
        self.applyTwoQubitOperation('cx', q, r)


cudaq.mpi.initialize()
NQPUS = cudaq.mpi.num_ranks()
RANK = cudaq.mpi.rank()
NUM_GLOBAL_QUBITS = 8
NUM_LOCAL_QUBITS = NUM_GLOBAL_QUBITS // NQPUS

qubit_map = {
    globalId: k
    for k, globalId in enumerate(
        range(RANK * (NUM_LOCAL_QUBITS),
              RANK * (NUM_LOCAL_QUBITS) + NUM_LOCAL_QUBITS))
}
r = 0
rank_map = {}
for i in range(NUM_GLOBAL_QUBITS):
    if i != 0 and i % NUM_LOCAL_QUBITS == 0:
        r = r + 1
    rank_map[i] = r

print(rank_map)
print(RANK, NQPUS, NUM_LOCAL_QUBITS, qubit_map)

qpu = QPU(RANK, qubit_map, rank_map)

# Gates operate on global qubits
qpu.h(0)
qpu.h(3)  # should sync with qpu 1 with numranks = 2

print(f'End {RANK}:\n{qpu.tensors}')

# qpu.cnot(0, 1)
# qpu.cnot(2,3)

print(f'End CNOT {RANK}:\n{qpu.tensors}')

qpu.cnot(5, 6)

print(f'End CNOT2 {RANK}:\n{qpu.tensors}')



cudaq.mpi.finalize()
