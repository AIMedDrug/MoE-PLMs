"""User-defined independent Optuna search dimensions, validated before training."""
import json
import math
from pathlib import Path

INTEGER_PARAMS = {'d_model', 'hidden_dim_mlp', 'num_experts', 'top_k', 'batch_size'}
PARAMS = INTEGER_PARAMS | {'lr', 'weight_decay', 'dropout'}


def validate_space(space):
    if not isinstance(space, dict) or set(space) != PARAMS:
        raise ValueError(f'搜索配置必须包含且仅包含 {sorted(PARAMS)}')
    bounds = {}
    for name, spec in space.items():
        integer = name in INTEGER_PARAMS
        if isinstance(spec, list):
            if not spec or len({str(v) for v in spec}) != len(spec):
                raise ValueError(f'{name}: 候选列表不能为空或包含重复值')
            values = spec
        elif isinstance(spec, dict):
            expected = 'int' if integer else 'float'
            if (set(spec) - {'type', 'low', 'high', 'step', 'log'} or
                    not {'type', 'low', 'high'}.issubset(spec) or spec['type'] != expected):
                raise ValueError(f'{name}: 范围需要 type={expected}、low、high，可选 step/log')
            values = [spec['low'], spec['high']]
            if not isinstance(spec.get('log', False), bool):
                raise ValueError(f'{name}: log 必须是布尔值')
        else:
            raise ValueError(f'{name}: 使用候选列表或范围对象，例如 [64, 96, 128]')
        for value in values:
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or (integer and not isinstance(value, int))):
                raise ValueError(f'{name}: 参数必须为有限数值，整数参数须填整数')
        lo, hi = min(values), max(values)
        if integer and lo < (2 if name in {'num_experts', 'batch_size'} else 1):
            raise ValueError(f'{name}: 参数过小；top_k须>=1，num_experts和batch_size须>=2')
        if (name == 'lr' and lo <= 0) or (name == 'weight_decay' and lo < 0) or (name == 'dropout' and not 0 <= lo <= hi < 1):
            raise ValueError(f'{name}: 数值范围无效')
        if name == 'd_model' and any(v % 4 for v in values):
            raise ValueError('d_model 必须为4的倍数，以匹配注意力头数')
        if isinstance(spec, dict):
            if spec['low'] > spec['high']:
                raise ValueError(f'{name}: low不能大于high')
            step = spec.get('step', 1 if integer else None)
            if step is not None:
                if (isinstance(step, bool) or not isinstance(step, (int, float)) or
                        not math.isfinite(step) or step <= 0 or (integer and not isinstance(step, int))):
                    raise ValueError(f'{name}: step须为正数，整数范围须用整数步长')
                if not math.isclose((hi - lo) / step, round((hi - lo) / step), abs_tol=1e-8):
                    raise ValueError(f'{name}: high-low 必须能被 step 整除')
            if spec.get('log', False) and (lo <= 0 or ('step' in spec and (not integer or step != 1))):
                raise ValueError(f'{name}: 对数范围须为正数，不能与此步长组合')
            if name == 'd_model' and lo != hi and (spec.get('log', False) or step % 4):
                raise ValueError('d_model 的整数范围步长须为4的倍数，且不使用log采样')
        bounds[name] = (lo, hi)
    if bounds['top_k'][1] > bounds['num_experts'][0]:
        raise ValueError('top_k最大值不能超过num_experts最小值，以保证任意采样组合有效')
    return space


def load_space(path):
    return validate_space(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def sample_space(trial, space):
    values = {}
    for name, spec in space.items():
        if isinstance(spec, list):
            values[name] = trial.suggest_categorical(name, spec)
        else:
            kwargs = {key: value for key, value in spec.items() if key != 'type'}
            method = trial.suggest_int if spec['type'] == 'int' else trial.suggest_float
            values[name] = method(name, **kwargs)
    return values
