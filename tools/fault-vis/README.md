vis.py requires plotly

## Environment

bash env-setup.sh
source .faultvis-env/bin/activate
deactive


## TODO

uuid as well to see why we have faults on the same addresses at different times

visualize the migration between the CPU and GPU 

visualize the size of the faults as a block instead of a dot on a graph
 - make this bigger cause I can barely see this in the novis

use pfns instead of addresses

find cudaMallocManaged to give idea of allocations (starts and sizes) and faults by allocations

Use metrics as warnings for users along with coalescing
