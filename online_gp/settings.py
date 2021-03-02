from gpytorch.settings import _feature_flag, _value_context

class check_decomposition(_feature_flag):
    _state = False

class detach_interp_coeff(_feature_flag):
    _state = False