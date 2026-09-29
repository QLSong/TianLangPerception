"""TensorBoard 里每条 loss 单独成图。"""
import math

from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger import TensorboardLoggerHook


@HOOKS.register_module()
class FlatTensorboardLoggerHook(TensorboardLoggerHook):
    """不给标量加 train/ 前缀，非有限值不写入。

    TensorBoard 会把带 / 的 tag 收进同一个折叠组。组里一旦有 inf，
    整组曲线会被压成一个点。
    """

    def get_loggable_tags(self, runner, allow_scalar=True, allow_text=False,
                          add_mode=True, tags_to_skip=('time', 'data_time')):
        del add_mode
        tags = super().get_loggable_tags(
            runner,
            allow_scalar=allow_scalar,
            allow_text=allow_text,
            add_mode=False,
            tags_to_skip=tags_to_skip)
        kept = {}
        for key, val in tags.items():
            if isinstance(val, str):
                kept[key] = val
                continue
            try:
                number = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                kept[key] = val
        return kept
