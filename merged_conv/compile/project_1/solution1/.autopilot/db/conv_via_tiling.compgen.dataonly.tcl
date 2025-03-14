# This script segment is generated automatically by AutoPilot

set axilite_register_dict [dict create]
set port_control {
in_ch { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 16
	offset_end 23
}
out_ch { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 24
	offset_end 31
}
H { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 32
	offset_end 39
}
W { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 40
	offset_end 47
}
input_r { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 48
	offset_end 59
}
weight { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 60
	offset_end 71
}
bias { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 72
	offset_end 83
}
output_r { 
	dir I
	width 64
	depth 1
	mode ap_none
	offset 84
	offset_end 95
}
stride { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 96
	offset_end 103
}
pad { 
	dir I
	width 32
	depth 1
	mode ap_none
	offset 104
	offset_end 111
}
}
dict set axilite_register_dict control $port_control


