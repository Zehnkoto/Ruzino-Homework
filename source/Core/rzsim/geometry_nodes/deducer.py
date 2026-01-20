import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add ShapeSpaceSpectra-reproduction project to Python path
project_root = Path(r"C:\Users\Pengfei\WorkSpace\ShapeSpaceSpectra-reproduction")
sys.path.insert(0, str(project_root))

from util.task_save_load import load_task_from_path

# Global basis_set to be loaded once
_basis_set = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_current_model_name = None


def initialize_basis_set(model_name="mesh_growing_y_neg_20260119_015826"):
    """
    Initialize and load the basis set model

    Args:
        model_name: Name of the model checkpoint folder
    """
    global _basis_set, _device, _current_model_name

    # If already initialized with the same model, skip
    if _basis_set is not None and _current_model_name == model_name:
        return f"Basis set already initialized with model: {model_name}"

    # If different model requested, force re-initialization
    if _basis_set is not None and _current_model_name != model_name:
        _basis_set = None
        print(f"Switching from model '{_current_model_name}' to '{model_name}'")

    # Checkpoint path
    checkpoint_path = rf"C:\Users\Pengfei\WorkSpace\ShapeSpaceSpectra-reproduction\checkpoints\{model_name}"

    if not os.path.exists(checkpoint_path):
        return f"Error: Checkpoint path does not exist: {checkpoint_path}"

    # Change working directory to Ruzino root (where test.py runs successfully)
    ruzino_root = r"C:\Users\Pengfei\WorkSpace\Ruzino"
    original_cwd = os.getcwd()
    os.chdir(ruzino_root)

    try:
        # Load task from checkpoint folder
        task_path = os.path.join(checkpoint_path, "task")
        if not os.path.exists(task_path):
            return f"Error: Task folder not found in checkpoint: {task_path}"

        # Load basis set and configuration
        _basis_set, task_name, training_config = load_task_from_path(task_path)

        # Load network weights
        checkpoint_file = os.path.join(checkpoint_path, "network_params.pt")
        if not os.path.exists(checkpoint_file):
            return f"Error: Network parameters not found: {checkpoint_file}"

        _basis_set.load_weights(checkpoint_file, device=_device)
        _current_model_name = model_name

        return f"Initialized basis set: {task_name}, model={model_name}, {_basis_set.num_fields} eigenfunctions, device={_device}"
    except Exception as e:
        import traceback

        return f"Error initializing basis set: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


def run_inference_on_vertices(vertices_cuda, eigenfunction_idx=0, shape_code=None):
    """
    Run basis set inference on vertices

    Args:
        vertices_cuda: CUDA torch tensor of shape (N, 3) - vertex positions
        eigenfunction_idx: index of eigenfunction to use (default: 0)
        shape_code: optional shape code tensor, if None will use zero code

    Returns:
        CPU numpy array with inference results
    """
    global _basis_set, _device

    if _basis_set is None:
        init_result = initialize_basis_set()
        if "Error" in init_result:
            raise RuntimeError(init_result)

    # Ensure vertices are on the correct device and shape
    if vertices_cuda.ndim == 1:
        vertices_cuda = vertices_cuda.reshape(-1, 3)

    vertices_cuda = vertices_cuda.to(_device)
    num_vertices = vertices_cuda.shape[0]

    # Prepare shape code (use zero code if not provided)
    if shape_code is None:
        # Network expects 5D input: 3D position + 1D shape_code + 1D distance_field
        # So shape_code should be a single scalar value
        shape_code = torch.tensor([0.0], device=_device)
    else:
        # Ensure shape_code is a tensor on the correct device
        if not isinstance(shape_code, torch.Tensor):
            shape_code = torch.tensor([float(shape_code)], device=_device)
        else:
            shape_code = shape_code.to(_device)

    # Clamp eigenfunction index
    eigenfunction_idx = max(0, min(eigenfunction_idx, _basis_set.num_fields - 1))

    # Run inference
    with torch.no_grad():
        output = _basis_set.inference(
            i=eigenfunction_idx,
            shape_code=shape_code,
            sample=vertices_cuda,
            device=_device,
        )

    # Convert to CPU numpy array and squeeze to 1D
    output_cpu = output.squeeze().cpu().numpy()

    # Log statistics
    print(
        f"Inference completed: {num_vertices} vertices, eigenfunction {eigenfunction_idx}"
    )
    print(f"Output range: [{float(output.min()):.4f}, {float(output.max()):.4f}]")
    print(f"Output mean: {float(output.mean()):.4f}, std: {float(output.std()):.4f}")

    return output_cpu


def get_cuda_tensor():
    """
    Returns a CUDA torch tensor of length 10
    """
    if torch.cuda.is_available():
        tensor = torch.arange(10, dtype=torch.float32, device="cuda")
        print(f"Python: Created CUDA tensor with shape {tensor.shape}")
        return tensor
    else:
        print("Warning: CUDA not available, returning CPU tensor")
        tensor = torch.arange(10, dtype=torch.float32)
        return tensor


def process_vertices_cuda(vertices_cuda):
    """
    Process geometry vertices on GPU
    vertices_cuda: CUDA torch tensor of shape (N*3,) or (N, 3)
    Returns: formatted string with vertex information
    """

    # Reshape if flat array
    if vertices_cuda.ndim == 1:
        vertices_cuda = vertices_cuda.reshape(-1, 3)

    num_vertices = vertices_cuda.shape[0]

    # Perform GPU operations
    # Example: compute bounding box
    min_vals = vertices_cuda.min(dim=0).values
    max_vals = vertices_cuda.max(dim=0).values
    center = vertices_cuda.mean(dim=0)

    # Copy first few vertices to CPU for display
    num_to_print = min(5, num_vertices)
    vertices_cpu = vertices_cuda[:num_to_print].cpu().numpy()

    lines = []
    lines.append(f"Received {num_vertices} vertices on CUDA")
    lines.append(f"Vertices shape: {vertices_cuda.shape}")
    lines.append(f"Device: {vertices_cuda.device}")
    lines.append(
        f"Bounding box: min=({min_vals[0]:.4f}, {min_vals[1]:.4f}, {min_vals[2]:.4f})"
    )
    lines.append(
        f"              max=({max_vals[0]:.4f}, {max_vals[1]:.4f}, {max_vals[2]:.4f})"
    )
    lines.append(f"Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    lines.append(f"First {num_to_print} vertices:")
    for i in range(num_to_print):
        lines.append(
            f"  [{i}] ({vertices_cpu[i][0]:.4f}, {vertices_cpu[i][1]:.4f}, {vertices_cpu[i][2]:.4f})"
        )

    return "\n".join(lines)
