#!/bin/bash

# Array to store binary names
binaries=()

# Find all executable files in the current directory
for file in *; do
    if [[ -x "$file" && -f "$file" ]]; then
        binaries+=("$file")
    fi
done
echo "Found ${#binaries[@]} binaries"
# Check if any binaries were found
if [ ${#binaries[@]} -eq 0 ]; then
    echo "No executable binaries found in the current directory."
    exit 1
fi

# Run each binary 5 times and redirect output to separate files
for binary in "${binaries[@]}"; do
    output_file="${binary}_output.txt"
    
    # Clear the output file if it already exists
    > "$output_file"
    
    for i in {1..5}; do
        echo "Run $i of $binary:" >> "$output_file"
        ./"$binary" >> "$output_file" 2>&1
        echo "" >> "$output_file"
    done
    
    echo "Output for $binary has been saved to $output_file"
done

echo "All binaries have been executed 5 times each, and their outputs have been saved."