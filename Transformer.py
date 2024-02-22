# Given architectural parameters
max_input_seq_length = 64
max_output_seq_length = 128
input_vocab_size = 10000
output_vocab_size = 5000
embedding_size = 256
num_attention_heads = 4
embedding_size_per_head = 32
num_encoder_blocks = 4
num_decoder_blocks = 4

# Assuming batch size of 1
batch_size = 1

# 1. Total number of self-attention blocks
total_self_attention_blocks = num_encoder_blocks + num_decoder_blocks

# 2. Total number of skip connections
total_skip_connections = 2*num_encoder_blocks+3*num_decoder_blocks

# 3. Total number of elements in the output tensor of the last encoder block
last_encoder_output_size = embedding_size * max_input_seq_length

# 4. Total number of elements in the output tensor of scaled dot product similarity
attention_weight_matrix_size = max_input_seq_length * embedding_size_per_head

# 5. Number of output units of the final fully connected layer
final_fc_output_units = output_vocab_size

# 6. Total number of weights in a single multi-head self-attention module
total_attention_weights = embedding_size*num_attention_heads*embedding_size_per_head*3 + embedding_size*embedding_size


# Print results
print("1. Total number of self-attention blocks:", total_self_attention_blocks)
print("2. Total number of skip connections:", total_skip_connections)
print("3. Total number of elements in the output tensor of the last encoder block:", last_encoder_output_size)
print("4. Total number of elements in the output tensor of scaled dot product similarity:", attention_weight_matrix_size)
print("5. Number of output units of the final fully connected layer:", final_fc_output_units)
print("6. Total number of weights in a single multi-head self-attention module:", total_attention_weights)
