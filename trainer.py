import logging

from torch.utils.tensorboard import SummaryWriter

from clearml import Task

def main(rank, cl_task_id):
    print(f'HELLOO IN MAIN from rank {rank}')
    logger = logging.getLogger(__name__)

    cl_task = Task.get_task(task_id=cl_task_id)
    print(f'Got Task {cl_task}, {cl_task.task_id}')

    logger.warning(f"Logging some stuff in rank {rank} BEFORE getting current_task")
    print(f'Printing some stuff in rank {rank} BEFORE getting current_task')

    if rank == 0:
        writer = SummaryWriter('tb')
        writer.add_scalar('mAP (before)', 0, 0)
        writer.add_scalar('mAP (before)', 0.5, 50)
        writer.add_scalar('mAP (before)', 1.0, 100)

    current_task = Task.current_task()
    print(f'Got current Task {current_task}, {current_task.task_id}')
    logger.warning(f"Logging some stuff in rank {rank} AFT getting current_task")
    print(f'Printing some stuff in rank {rank} AFT getting current_task')
    
    if rank == 0:
        writer = SummaryWriter('tb')
        writer.add_scalar('mAP (aft)', 0, 0)
        writer.add_scalar('mAP (aft)', 0.5, 50)
        writer.add_scalar('mAP (aft)', 1.0, 100)
