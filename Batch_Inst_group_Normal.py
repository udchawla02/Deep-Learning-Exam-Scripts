# Given values
number_of_samples = 100
channel_width = 30
spatial_resolution = 24
num_groups = 5  # You can choose the number of groups based on your configuration


# Parameters per channel for Batch Normalization
parameters_per_channel_bn = 2  # Scale (gamma) and Shift (beta)
total_parameters_bn = parameters_per_channel_bn * channel_width

# Parameters per channel for Instance Normalization
parameters_per_channel_in = 2  # Scale (gamma) and Shift (beta)
total_parameters_in = parameters_per_channel_in * channel_width

# Parameters for Group Normalization
parameters_per_group_gn = 2  # Scale (gamma) and Shift (beta)
total_parameters_gn = parameters_per_group_gn * num_groups

# Parameters per channel for Layer Normalization
parameters_per_channel_ln = 2  # Scale (gamma) and Shift (beta)
total_parameters_ln = parameters_per_channel_ln * channel_width

print("Total parameters for Batch Normalization layer:", total_parameters_bn)
print("Total parameters for Instance Normalization layer:", total_parameters_in)
print("Total parameters for Group Normalization layer:", total_parameters_gn)
print("Total parameters for Layer Normalization layer:", total_parameters_ln)
