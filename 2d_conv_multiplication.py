# Given values
input_channels = 3
output_channels = 10
kernel_size = (5, 5)
stride = 5
padding = 0
dilation = 2
input_size = (25, 25)
#If not given take 1 as number of images
No_of_images= 7

# Calculate output size for dilation

#output_size = (
#    (input_size[0] + 2 * padding - dilation * (kernel_size[0] - 1) - 1) // stride + 1,
#    (input_size[1] + 2 * padding - dilation * (kernel_size[1] - 1) - 1) // stride + 1
#)

#Calculate output size without dilation

output_size=(
        (input_size[0] - kernel_size[0] + 2 * padding) // stride + 1,
        (input_size[1] - kernel_size[1] + 2 * padding) // stride + 1
)
# Calculate multiplication operations per image

multiplications = input_channels * output_channels * kernel_size[0] * kernel_size[1] * output_size[0] * output_size[1]

# Calculate multiplication operations to number of image

print("Multiplication operations in the forward pass:", multiplications * No_of_images)
