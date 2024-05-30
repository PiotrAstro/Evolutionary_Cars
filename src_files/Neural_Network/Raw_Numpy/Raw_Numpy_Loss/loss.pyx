import cython
import numpy as np
from src_files.MyMath.cython_debug_helper import cython_debug_call
cpdef float[:, ::1] calculate_loss(float[:, ::1] pred, float[:, ::1] label, losses: list[tuple[str, int]]):
    cdef float[:, ::1] grad = np.empty((pred.shape[0], pred.shape[1]), dtype=np.float32)
    cdef float[:, ::1] pred_part
    cdef float[:, ::1] label_part
    cdef float[:, ::1] grad_part
    cdef int prev_loss_idx = 0
    cdef int new_loss_idx = 0

    for loss_name, loss_idx in losses:
        new_loss_idx = prev_loss_idx + loss_idx
        pred_part = pred[:, prev_loss_idx:new_loss_idx]
        label_part = label[:, prev_loss_idx:new_loss_idx]
        grad_part = grad[:, prev_loss_idx:new_loss_idx]

        if loss_name == 'MSE':
            with nogil:
                MSE_grad(pred_part, label_part, grad_part)
        elif loss_name == 'Cross_Entropy':
            with nogil:
                Cross_Entropy_grad(pred_part, label_part, grad_part)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
        prev_loss_idx = new_loss_idx

    return grad

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int MSE_grad(float[:, ::1] pred, float[:, ::1] label, float[:, ::1] grad) noexcept nogil:
    cdef int row, col
    cdef int num_rows = pred.shape[0]
    cdef int num_cols = pred.shape[1]
    for row in range(num_rows):
        for col in range(num_cols):
            # MSE gradient
            grad[row, col] = 2 * (pred[row, col] - label[row, col])
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int Cross_Entropy_grad(float[:, ::1] pred, float[:, ::1] label, float[:, ::1] grad) noexcept nogil:
    cdef int row, col
    cdef int num_rows = pred.shape[0]
    cdef int num_cols = pred.shape[1]
    for row in range(num_rows):
        for col in range(num_cols):
            # Cross Entropy gradient
            grad[row, col] = -label[row, col] / (pred[row, col] + 1e-8)

    # with gil:
    #     cython_debug_call({
    #         "pred": np.array(pred),
    #         "label": np.array(label),
    #         "grad": np.array(grad),
    #     })

    return 0