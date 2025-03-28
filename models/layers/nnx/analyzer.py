from flax import nnx


class Analyzer:
    def __init__(self):
        pass

    def __call__(self, model: nnx.Module):
        layer_info = []

        for name, module in model.iter_modules():
            # Check if the module contains 'layers' in the name
            if "layers" in name:
                # Extract the index from the tuple
                idx = name[2]

                # Store the layer type and parameters
                layer_data = {"index": idx, "type": type(module).__name__, "params": {}}

                # Extract parameters from the module's attributes
                for attr_name, attr_value in module.__dict__.items():
                    if not attr_name.startswith("__") and "state" not in attr_name:
                        layer_data["params"][attr_name] = attr_value

                layer_info.append(layer_data)

        # Output the list of layer info
        return tuple(layer_info)
