import onnxruntime as rt

# List all EPs that this build of onnxruntime knows about:
print("Available in this build:\n", rt.get_available_providers())