import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=str, required=True)
parser.add_argument("--data_name", type=str, required=True) # 例如 Istanbul_PO1
parser.add_argument("--world_size", type=int, default=4)
args = parser.parse_args()

all_sequences = []
t_max = None

base_dir = f'./data/{args.data_name}'

for rank in range(args.world_size):
    filename = f'{base_dir}/{args.data_name}_{args.run_id}_generated_part{rank}.pkl'
    if not os.path.exists(filename):
        print(f"Error: Missing file {filename}")
        continue
        
    print(f"Loading {filename}...")
    data = torch.load(filename)
    all_sequences.extend(data['sequences'])
    
    # 假设 t_max 是一样的，取第一个即可
    if t_max is None:
        t_max = data['t_max']

# 保存合并后的文件
final_filename = f'{base_dir}/{args.data_name}_{args.run_id}_generated.pkl'
final_data = {'sequences': all_sequences, 't_max': t_max}
torch.save(final_data, final_filename)

print(f"Successfully merged {len(all_sequences)} sequences into {final_filename}")

# 删除分片文件
for rank in range(args.world_size):
    os.remove(f'{base_dir}/{args.data_name}_{args.run_id}_generated_part{rank}.pkl')