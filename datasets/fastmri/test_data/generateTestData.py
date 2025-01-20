import h5py

with h5py.File('./file1000000.h5', 'r') as hf_source:
    rss_data = hf_source['reconstruction_rss'][:]

with h5py.File('test_knee_data.h5', 'w') as hf_target:
    hf_target.create_dataset('tstOrg', data=rss_data)

print("数据已成功转移到新文件，并重命名为 'tstOrg'")