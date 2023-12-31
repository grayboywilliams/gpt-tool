# main
app_name = "gpt-tool"
host = "0.0.0.0"
port = 5000

# model
name = 'name'
data_source = 'data_source'

# architecture params
ctx_length = 'ctx_length'
num_dim = 'num_dim'
num_head = 'num_head'
num_layer = 'num_layer'

# training
batch_size = 'batch_size'
num_batch = 'num_batch'
learning_rate = 'learning_rate'
dropout = 'dropout'
eval_interval = 'eval_interval'
eval_size = 'eval_size'
val_split = 'val_split'
test_split = 'test_split'

# pytorch
torch_seed = 'torch_seed'
cuda = 'cuda'
cpu = 'cpu'

# request args
name_arg = 'name'
prompt_arg = 'prompt'
length_arg = 'length'
temp_arg = 'temp'

# data sets
train = 'train'
val = 'val'
test = 'test'

# http methods
POST = 'POST'
GET = 'GET'

# directories
checkpoints = '../checkpoints'
configs = '../configs'
temp = 'temp'