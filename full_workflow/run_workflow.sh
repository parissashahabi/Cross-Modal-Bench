#!/bin/bash

# Interactive script for running the workflow locally
# This script activates the appropriate environment and runs workflow steps

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="${SCRIPT_DIR}"

echo "========================================"
echo "Full Workflow Execution (Interactive)"
echo "========================================"
echo "Workflow directory: ${WORKFLOW_DIR}"
echo "Start time: $(date)"
echo ""

# Change to workflow directory
cd "${WORKFLOW_DIR}"

# Check if config file exists
if [ ! -f "config.json" ]; then
    echo "ERROR: config.json not found in ${WORKFLOW_DIR}"
    exit 1
fi

echo "Using configuration: config.json"
echo ""

# Parse command line arguments
START_STEP=1
END_STEP=6
CONFIG_FILE="config.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --start)
            START_STEP="$2"
            shift 2
            ;;
        --end)
            END_STEP="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE    Configuration file (default: config.json)"
            echo "  --start STEP     First step to run (1-5, default: 1)"
            echo "  --end STEP       Last step to run (1-5, default: 5)"
            echo "  --help           Show this help message"
            echo ""
            echo "Steps:"
            echo "  1. Segmentation"
            echo "  2. Scene Graph Generation"
            echo "  3. Attribute Generation"
            echo "  4. Graph Matching"
            echo "  5. Visualization"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running steps ${START_STEP} to ${END_STEP}"
echo ""

# Initialize conda and poetry
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
fi

echo "Note: Different steps use different environments:"
echo "  - Step 1 (kmax segmentation): conda env kmax_deeplab"
echo "  - Steps 2-3 (scene graphs, attributes): poetry env"
echo "  - Steps 4-5 (matching, visualization): conda env for graph matching"
echo "  The workflow script handles environment switching automatically."
echo ""

# Run the workflow
python run_full_workflow.py \
    --config "${CONFIG_FILE}" \
    --workflow-dir "${WORKFLOW_DIR}" \
    --start ${START_STEP} \
    --end ${END_STEP}

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Workflow Completed"
echo "========================================"
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo ""

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Workflow completed successfully"
    echo ""
    echo "Output directories:"
    # Step 1: segmentation
    if [ -d "${WORKFLOW_DIR}/1_segmentation" ]; then
        file_count=$(find "${WORKFLOW_DIR}/1_segmentation" -type f | wc -l)
        echo "  1_segmentation: ${file_count} files"
    fi
    # Step 2: scene graphs (only GT/PT)
    for dir in 2_scene_graphs_gt 2_scene_graphs_pt; do
        if [ -d "${WORKFLOW_DIR}/${dir}" ]; then
            file_count=$(find "${WORKFLOW_DIR}/${dir}" -type f | wc -l)
            echo "  ${dir}: ${file_count} files"
        fi
    done
    # Step 3: graph merge (only GT/PT)
    for dir in 3_graph_merge_gt 3_graph_merge_pt; do
        if [ -d "${WORKFLOW_DIR}/${dir}" ]; then
            file_count=$(find "${WORKFLOW_DIR}/${dir}" -type f | wc -l)
            echo "  ${dir}: ${file_count} files"
        fi
    done
    # Step 4: attributes (only GT/PT)
    for dir in 4_attributes_gt 4_attributes_pt; do
        if [ -d "${WORKFLOW_DIR}/${dir}" ]; then
            file_count=$(find "${WORKFLOW_DIR}/${dir}" -type f | wc -l)
            echo "  ${dir}: ${file_count} files"
        fi
    done
    # Step 5: matching
    if [ -d "${WORKFLOW_DIR}/5_matching" ]; then
        file_count=$(find "${WORKFLOW_DIR}/5_matching" -type f | wc -l)
        echo "  5_matching: ${file_count} files"
    fi
    # Step 6: visualizations
    if [ -d "${WORKFLOW_DIR}/6_visualizations" ]; then
        file_count=$(find "${WORKFLOW_DIR}/6_visualizations" -type f | wc -l)
        echo "  6_visualizations: ${file_count} files"
    fi
else
    echo "✗ Workflow failed with exit code ${EXIT_CODE}"
    echo "Check logs in: ${WORKFLOW_DIR}/logs/"
fi

exit ${EXIT_CODE}
