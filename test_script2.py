from utils.data_loading import load_cache_output_data

output = load_cache_output_data("output/cache/science&2021-10-01_2022-09-30&nmf&20.obj")

print(output.metrics_dict)