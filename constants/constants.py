# main
app_name = "gpt-tool"
host = "0.0.0.0"
port = 5000

# model
name = 'name'
data_source = 'data_source'

# architecture params
ctx_length = 'ctx_length'
batch_size = 'batch_size'
num_dim = 'num_dim'
num_head = 'num_head'
num_layer = 'num_layer'
head_size = 'head_size'

# training
dropout = 'dropout'
learning_rate = 'learning_rate'
num_batch = 'num_batch'
eval_interval = 'eval_interval'
eval_iters = 'eval_iters'
val_split = 'val_split'
test_split = 'test_split'

# pytorch
torch_seed = 'torch_seed'
cuda = 'cuda'
cpu = 'cpu'

# request args
prompt = 'prompt'
length = 'length'
temp = 'temp'

# data sets
train = 'train'
val = 'val'
test = 'test'

# http methods
POST = 'POST'
GET = 'GET'
