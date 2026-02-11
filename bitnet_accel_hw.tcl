# TCL File Generated for BitNet Accelerator Platform Designer Component
# BitNet b1.58 DSP-Free Inference Accelerator
# Avalon-MM Slave (HPS control) + Avalon-MM Master (DDR3 weight reads)

package require -exact qsys 16.1

#
# module bitnet_accel
#
set_module_property DESCRIPTION "BitNet b1.58 DSP-Free Inference Accelerator"
set_module_property NAME bitnet_accel
set_module_property VERSION 1.0
set_module_property INTERNAL false
set_module_property OPAQUE_ADDRESS_MAP true
set_module_property AUTHOR "BitNet"
set_module_property DISPLAY_NAME "BitNet Accelerator"
set_module_property INSTANTIATE_IN_SYSTEM_MODULE true
set_module_property EDITABLE true
set_module_property REPORT_TO_TALKBACK false
set_module_property ALLOW_GREYBOX_GENERATION false
set_module_property REPORT_HIERARCHY false


#
# file sets
#
add_fileset QUARTUS_SYNTH QUARTUS_SYNTH "" ""
set_fileset_property QUARTUS_SYNTH TOP_LEVEL BitNetAccelerator
set_fileset_property QUARTUS_SYNTH ENABLE_RELATIVE_INCLUDE_PATHS false
set_fileset_property QUARTUS_SYNTH ENABLE_FILE_OVERWRITE_MODE false
add_fileset_file BitNetAccelerator.sv SYSTEM_VERILOG PATH bitnet/chisel/generated/BitNetAccelerator.sv


#
# parameters
#


#
# display items
#


#
# connection point clock
#
add_interface clock clock end
set_interface_property clock clockRate 0
set_interface_property clock ENABLED true
set_interface_property clock EXPORT_OF ""
set_interface_property clock PORT_NAME_MAP ""
set_interface_property clock CMSIS_SVD_VARIABLES ""
set_interface_property clock SVD_ADDRESS_GROUP ""

add_interface_port clock clock clk Input 1


#
# connection point reset
#
add_interface reset reset end
set_interface_property reset associatedClock clock
set_interface_property reset synchronousEdges DEASSERT
set_interface_property reset ENABLED true
set_interface_property reset EXPORT_OF ""
set_interface_property reset PORT_NAME_MAP ""
set_interface_property reset CMSIS_SVD_VARIABLES ""
set_interface_property reset SVD_ADDRESS_GROUP ""

add_interface_port reset reset reset Input 1


#
# connection point avs_slave — Avalon-MM Slave (HPS control/status, activation writes, result reads)
#
add_interface avs_slave avalon end
set_interface_property avs_slave addressUnits SYMBOLS
set_interface_property avs_slave associatedClock clock
set_interface_property avs_slave associatedReset reset
set_interface_property avs_slave bitsPerSymbol 8
set_interface_property avs_slave burstOnBurstBoundariesOnly false
set_interface_property avs_slave burstcountUnits WORDS
set_interface_property avs_slave explicitAddressSpan 0
set_interface_property avs_slave holdTime 0
set_interface_property avs_slave linewrapBursts false
set_interface_property avs_slave maximumPendingReadTransactions 0
set_interface_property avs_slave maximumPendingWriteTransactions 0
set_interface_property avs_slave readLatency 1
set_interface_property avs_slave readWaitTime 0
set_interface_property avs_slave setupTime 0
set_interface_property avs_slave timingUnits Cycles
set_interface_property avs_slave writeWaitTime 0
set_interface_property avs_slave ENABLED true
set_interface_property avs_slave EXPORT_OF ""
set_interface_property avs_slave PORT_NAME_MAP ""
set_interface_property avs_slave CMSIS_SVD_VARIABLES ""
set_interface_property avs_slave SVD_ADDRESS_GROUP ""

add_interface_port avs_slave io_slave_address address Input 14
add_interface_port avs_slave io_slave_read read Input 1
add_interface_port avs_slave io_slave_write write Input 1
add_interface_port avs_slave io_slave_writedata writedata Input 32
add_interface_port avs_slave io_slave_readdata readdata Output 32

set_interface_assignment avs_slave embeddedsw.configuration.isFlash 0
set_interface_assignment avs_slave embeddedsw.configuration.isMemoryDevice 0
set_interface_assignment avs_slave embeddedsw.configuration.isNonVolatileStorage 0
set_interface_assignment avs_slave embeddedsw.configuration.isPrintableDevice 0


#
# connection point avm_master — Avalon-MM Master (DDR3 weight streaming, 128-bit)
#
add_interface avm_master avalon start
set_interface_property avm_master addressUnits SYMBOLS
set_interface_property avm_master associatedClock clock
set_interface_property avm_master associatedReset reset
set_interface_property avm_master bitsPerSymbol 8
set_interface_property avm_master burstOnBurstBoundariesOnly false
set_interface_property avm_master burstcountUnits WORDS
set_interface_property avm_master doStreamReads false
set_interface_property avm_master doStreamWrites false
set_interface_property avm_master holdTime 0
set_interface_property avm_master linewrapBursts false
set_interface_property avm_master maximumPendingReadTransactions 1
set_interface_property avm_master maximumPendingWriteTransactions 0
set_interface_property avm_master readLatency 0
set_interface_property avm_master readWaitTime 1
set_interface_property avm_master setupTime 0
set_interface_property avm_master timingUnits Cycles
set_interface_property avm_master writeWaitTime 0
set_interface_property avm_master ENABLED true
set_interface_property avm_master EXPORT_OF ""
set_interface_property avm_master PORT_NAME_MAP ""
set_interface_property avm_master CMSIS_SVD_VARIABLES ""
set_interface_property avm_master SVD_ADDRESS_GROUP ""

add_interface_port avm_master io_master_address address Output 32
add_interface_port avm_master io_master_read read Output 1
add_interface_port avm_master io_master_readdata readdata Input 128
add_interface_port avm_master io_master_waitrequest waitrequest Input 1
add_interface_port avm_master io_master_readdatavalid readdatavalid Input 1


#
# Device tree generation
#
set_module_assignment embeddedsw.dts.vendor "bitnet"
set_module_assignment embeddedsw.dts.compatible "bitnet,accelerator-1.0"
set_module_assignment embeddedsw.dts.group "bitnet"
