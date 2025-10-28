## Proposed CUDA-Q Refactor Task List

Here we enumerate expected tasks for the full CUDA-Q runtime refactor. 

We anticipate that this work can be done in a separate `rt` folder, alongside the existing 
`runtime` folder. Features can be added there incrementally, and once the full port is complete, we can swap out the original `runtime` folder with this new one. 

* `launch.h`
	* description: implement `launch()` overloads that dispatch to correct execution policy `launch_impl` via hidden friend ADL pattern.
	* Launch API 
* `qpu.h`
	* description: implement `qpu` base type with start-up configuration via user-specified (or nvq++ flag-specified) initial options.
	* QPU CRTP Mixin Pattern
* execution traits 
	* description: introduce initial foundational traits that enable compile-time dispatch of execution policy launch for local or remote scenarios. 
	* local (returns `ExecutionPolicy::result_type)
	* remote (returns `job<ExecutionPolicy::result_type>)
* `simulation_trait 
	* description: define an API for local simulation, expose it as a trait that subtypes can opt-into. 
	* simulation api, `get_state` (state and simulation_state), and noise modeling
	* `operation_metadata` type encodes useful name, parameters information for subtypes
* concrete simulation qpus (`local_trait`). all should be stand-alone implementations
	* description: start building out concrete simulation implementations. goal here is to drop complex, unified CircuitSimulator and replace with policy-aware individual implementations that can opt-into simulation-specific traits. 
	* custatevec (`cudaq::simulator::gpu::state_vector)
	* qpp cpu (`cudaq::simulator::cpu::state_vector)
	* qpp dm (`cudaq::simulator::cpu::density_matrix)
	* stim (`cudaq::simulator::cpu::stim` or `cudaq::simulator::cpu::clifford)
	* Others (not immediately needed as we build up)
		* all tensornet (`cudaq::simulator::gpu::mps`, `cudaq::simulator::gpu::tensornet`)
		* cusvsim (`cudaq::simulator::mgpu::state_vector`)
		* dynamics (`cudaq::simulator::gpu::dynamics`)
		* Photonics qudit simulator 
* default QPU configuration sub-system
	* description: we need a mechanism for app-startup configuration of user-specified QPUs (specified via command line --target). We also need to provide a runtime-swappable QPU that supports Python `set_target` use cases. 
	* --target, and python set_target support
* thread_local trait type-erasure sub-system
	* description: it is likely that type-erasure will prove critical in interfacing holistic kernel code to local trait QPUs. there is likely a common pattern here we can distill out.  
* QIS system delegating to type-erased simulation_trait
	* description: this is likely an update of `qubit_qis.h` to use the type-erased simulation trait QPU instead of `getExecutionManager()`
	* `qubit`, `qvector`, `qarray`, `qview`, `state`, `pauli_word`, modifiers, kernel_utils 
	* single qubit ops, rotations, ctrl/adj modifiers 
	* `exp_pauli 
	* measurement
	* `control()` / `adjoint()`
	* `apply_noise 
* local execution policies
	* description: update our current launcher implementations to use the new unified launch with specified execution policies. 
	* sample and sample-async 
	* observe, observe-async 
	* observe MQPU
	* run, run-async 
	* evolve 
	* explicit measurements and bit_table
* remote execution policies 
	* description: Here we have lots to figure out about MLIR coupling to the runtime (hit lots of issues in the past, hence cudaq-mlir-runtime library, is there a better way?)
	* sample (no need for async, returns job handle)
	* observe
	* run (question on return types vs explicit output logging, Eric's MR 38)
	* evolve 
	* explicit measurements 
	* Concrete Remote Trait QPUs 
		* Quantinuum 
		* Anyon 
		* Braket 
		* Infleqtion 
		* IonQ
		* IQM
		* OQC
		* QCI
		* QM 
		* ORCA
		* Pasqal 
		* QuEra
		* Fermioniq
* runtime kernel operational semantic checking 
	* e.g. is this a valid sample kernel, is this a valid observe kernel, etc.
* Runtime Circuit Drawing / Tracing 
* NVQIR Update to use type-erased simulation trait QPU

Things that should not be retained (but instead have been moved or should be moved to CUDA-QX)
* VQE (solvers)
* Optimizers (solvers)
* Gradients (solvers)
* DEM / MSM Execution Context work (qec) 
* kernel_builder dropped
* domains sub-folder (solvers)

Expected Breaking Changes 
* No Remote State Handling
* Kernel Builder removed