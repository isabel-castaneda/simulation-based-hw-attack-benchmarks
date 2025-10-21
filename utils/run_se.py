# run_se.py: SE-mode simulation with L1+L2 cache hierarchy and configurable binary.

from gem5.components.cachehierarchies.classic.private_l1_private_l2_cache_hierarchy import (
    PrivateL1PrivateL2CacheHierarchy,
)
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.boards.simple_board import SimpleBoard
from gem5.isas import ISA
from gem5.simulate.simulator import Simulator
from gem5.resources.resource import BinaryResource

import sys

# Get binary path from command line or use default
binary_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/nas/3/isabelc/gem5/code/loop_sum"

# Define processor: 1 TIMING core, RISCV ISA
processor = SimpleProcessor(cpu_type=CPUTypes.TIMING, isa=ISA.RISCV, num_cores=1)

# Define memory
memory = SingleChannelDDR3_1600("2GiB")

# Define cache hierarchy with L1I, L1D and L2
cache_hierarchy = PrivateL1PrivateL2CacheHierarchy(
    l1d_size="16kB", l1i_size="16kB", l2_size="256kB"
)

# Assemble board
board = SimpleBoard(
    clk_freq="1GHz", processor=processor, memory=memory, cache_hierarchy=cache_hierarchy
)

# Set the binary workload
board.set_se_binary_workload(BinaryResource(binary_path))

# Run the simulation
simulator = Simulator(board=board)
simulator.run()

