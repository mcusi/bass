import torch

# CUDA Robust - code to guard inference against CUDA errors

def cuda_reset():
    torch.set_default_tensor_type(torch.cuda.FloatTensor) 

def cuda_robust(f, on_cuda_error=None):
    try:
        return f()
    except RuntimeError as e:
        if "cuda" in str(e).lower() or "cufft" in str(e).lower():
            print("Crashed: ", str(e))
            print("Trying on cpu")
            torch.cuda.empty_cache()
            torch.backends.cuda.cufft_plan_cache.clear()
            torch.set_default_tensor_type(torch.FloatTensor)
            if on_cuda_error is not None:
                on_cuda_error(lambda x: x.to('cpu'))
            return f()
        else:
            raise e

def get_exclude_nodes(gres):
    """ Specify nodes to ignore on slurm scheduler"""
    return "node060,node062,node064,node059,node063"