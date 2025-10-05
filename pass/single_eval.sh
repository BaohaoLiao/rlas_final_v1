
#!/bin/bash

# Configuration
base_output_dir="data/gen_data_validate"
mkdir -p $base_output_dir
K=256
world_size=8

# Model and dataset arrays
models=('Qwen/Qwen2.5-Math-1.5B')


datasets=("weqweasdas/validate")

# Create base output directory
mkdir -p $base_output_dir

# Loop through models and datasets
for model_name in "${models[@]}"; do
    echo "Testing model: $model_name"
    
    for dataset in "${datasets[@]}"; do
        echo "Testing dataset: $dataset"
        
        # Create model/dataset specific output directory
        # Extract global_step_X/merged from the full path
        #model_step_dir=$(echo "$model_name" | sed 's|.*/\(global_step_[0-9]*/merged\)|\1|')
        output_dir="$base_output_dir/$dataset"
       # output_dir="$base_output_dir/qwen15b_step800_ada_balance_pass_rate_est_test_set/$dataset"
        
        mkdir -p "$output_dir"
        
        echo "Output directory: $output_dir"
       
        # Generate data in parallel
        echo "Starting parallel data generation..."
        # we use gpu 0,1,2,3,5,6,7,9
        gpu_list=(0 1 2 3 4 5 6 7)
        for idx in "${!gpu_list[@]}"; do
            gpu_id=${gpu_list[$idx]}
            CUDA_VISIBLE_DEVICES=$gpu_id python3 gen_data.py \
                --local_index $idx \
                --my_world_size $world_size \
                --model_name_or_path "$model_name" \
                --output_dir "$output_dir/" \
                --K $K \
                --dataset_name_or_path "$dataset" &
        done
        
        # Wait for all parallel processes to complete
        wait
        echo "Data generation completed."
        
        # Merge the generated data
        echo "Merging data..."
        
        python3 merge_data.py \
            --base_path "$output_dir/" \
            --output_dir "$output_dir/merged_data.jsonl" \
            --num_datasets $world_size
        
        
        
        # Compute scores
        echo "Computing scores..."
        python3 compute_score.py \
            --dataset_path "$output_dir/merged_data.jsonl" \
            --record_path "$output_dir/record.txt"
        
        if [ $? -ne 0 ]; then
            echo "Error: Failed to compute scores for $model_name on $dataset"
            continue
        fi
        
        echo "Completed evaluation for $model_name on $dataset"
        echo "Results saved to: $output_dir/record.txt"
        echo "----------------------------------------"
    done
done

echo "All evaluations completed!"
