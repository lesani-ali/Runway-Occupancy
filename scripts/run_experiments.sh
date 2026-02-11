#!/bin/bash

set -e
set -o pipefail

# Experiment runner script for testing different model configurations
# This script runs detection with different models and SAHI settings
#
# Usage: ./run_experiments.sh -s <source_video> [-c <config_file>]
#   -s: Path to input video file (required)
#   -c: Path to config file (optional, default: config/default.yaml)
#   -r: Path to results file (optional, default: experiment_results.txt)

# Default values
CONFIG_FILE="config/default.yaml"
SOURCE_VIDEO=""
RESULTS_FILE="experiment_results.txt"

# Parse command line arguments
while getopts "s:c:r:h" opt; do
    case $opt in
        s)
            SOURCE_VIDEO="$OPTARG"
            ;;
        c)
            CONFIG_FILE="$OPTARG"
            ;;
        r)
            RESULTS_FILE="$OPTARG"
            ;;
        h)
            echo "Usage: $0 -s <source_video> [-c <config_file>]"
            echo "  -s: Path to input video file (required)"
            echo "  -c: Path to config file (optional, default: config/default.yaml)"
            echo "  -r: Path to results file (optional, default: experiment_results.txt)"
            echo "Example: $0 -s ./data/takeoff.mp4 -c config/default.yaml -r experiment_results.txt"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "Use -h for help"
            exit 1
            ;;
    esac
done

# Check if source video is provided
if [ -z "$SOURCE_VIDEO" ]; then
    echo "Error: Source video path is required (-s flag)"
    echo "Usage: $0 -s <source_video> [-c <config_file> -r <results_file>]"
    echo "Use -h for help"
    exit 1
fi

# Validate that source video exists
if [ ! -f "$SOURCE_VIDEO" ]; then
    echo "Error: Source video file not found: $SOURCE_VIDEO"
    exit 1
fi

echo "Configuration:"
echo "  Source video: $SOURCE_VIDEO"
echo "  Config file: $CONFIG_FILE"
echo "  Results file: $RESULTS_FILE"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create results file with header
echo "Experiment Results - $(date)" > $RESULTS_FILE
echo "======================================" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Function to update YAML config
update_config() {
    local weight=$1
    local use_sahi=$2
    
    # Use sed to update the config file
    # Inplace substitution command: sed -i "s/pattern/replacement/" filename
    sed -i "s/^weights: .*/weights: \"$weight\"/" $CONFIG_FILE
    sed -i "s/^use_sahi: .*/use_sahi: $use_sahi/" $CONFIG_FILE
}

# Function to run experiment
run_experiment() {
    local weight=$1
    local use_sahi=$2
    local exp_num=$3
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment $exp_num/6${NC}"
    echo -e "${YELLOW}Model: $weight${NC}"
    echo -e "${YELLOW}SAHI: $use_sahi${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    # Update configj
    update_config "$weight" "$use_sahi"
    
    # Record start time
    start_time=$(date +%s)
    
    # Run the detection
    python src/main.py --config $CONFIG_FILE --source $SOURCE_VIDEO
    
    # Record end time
    end_time=$(date +%s)
    
    # Calculate duration
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    # Log results
    echo "" >> $RESULTS_FILE
    echo "Experiment $exp_num:" >> $RESULTS_FILE
    echo "  Model: $weight" >> $RESULTS_FILE
    echo "  SAHI: $use_sahi" >> $RESULTS_FILE
    echo "  Duration: ${minutes}m ${seconds}s (${duration}s total)" >> $RESULTS_FILE
    echo "  Completed at: $(date)" >> $RESULTS_FILE
    
    echo -e "${GREEN}✓ Completed in ${minutes}m ${seconds}s${NC}"
    echo ""
}

# Main execution
echo -e "${GREEN}Starting experiments...${NC}"
echo ""

# Experiment 1: fasterrcnn_resnet50_fpn, no SAHI
run_experiment "fasterrcnn_resnet50_fpn" "false" 1

# Experiment 2: fasterrcnn_resnet50_fpn, with SAHI
run_experiment "fasterrcnn_resnet50_fpn" "true" 2

# Experiment 3: fasterrcnn_resnet50_fpn_v2, no SAHI
run_experiment "fasterrcnn_resnet50_fpn_v2" "false" 3

# Experiment 4: fasterrcnn_resnet50_fpn_v2, with SAHI
run_experiment "fasterrcnn_resnet50_fpn_v2" "true" 4

# Experiment 5: retinanet_resnet50_fpn, no SAHI
run_experiment "retinanet_resnet50_fpn" "false" 5

# Experiment 6: retinanet_resnet50_fpn, with SAHI
run_experiment "retinanet_resnet50_fpn" "true" 6

# Final summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
cat $RESULTS_FILE
