"""Use Slurm's deadline when exported, with a conservative allocation-entry fallback."""
import os

def deadline():
 value=os.environ.get('SLURM_JOB_END_TIME')
 if value and value.isdigit():return int(value)
 return int(os.environ['EXPERIMENT_NODE_STARTED'])+int(os.environ['EXPERIMENT_WALL_SECONDS'])
