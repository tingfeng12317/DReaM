import os
import yaml
import sys


def _get_project_root():
    """从当前文件位置向上回溯到项目根目录"""
    # 本文件在 src/utils/，向上两级到项目根
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class Config:
    def __init__(self, yaml_path):
        self._yaml_path = os.path.abspath(yaml_path)
        self._project_root = _get_project_root()

        with open(self._yaml_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.dataset_name = self.cfg['dataset']['name']
        self.num_classes = self.cfg['dataset']['num_classes']

        # 输出根目录：项目根目录下的 outputs/
        self.output_root = os.path.join(self._project_root, 'outputs')
        os.makedirs(self.output_root, exist_ok=True)

        # 数据集专属输出目录：项目根目录下的 outputs/cifar100/
        self.dataset_output_dir = os.path.join(self.output_root, self.dataset_name)
        os.makedirs(self.dataset_output_dir, exist_ok=True)

        # 自动拼接所有中间路径（全部绝对路径）
        K = self.cfg['dream']['K']
        self.paths = {
            'topk': os.path.join(self.dataset_output_dir, 'topk_indices', f'k{K}.json'),
            'distribution': os.path.join(self.dataset_output_dir, 'distributions', f'k{K}_distributions.json'),
            'metrics': os.path.join(self.dataset_output_dir, 'metrics'),
            'grid_search': os.path.join(self.dataset_output_dir, 'grid_search'),
            'figures': os.path.join(self.dataset_output_dir, 'figures'),
        }

        # 允许 YAML 的 paths 段覆盖单个自动拼接的路径（output_root 除外）。
        # 用途：消融实验（随机头 / 不同 K）需要独立的 topk / distribution / metrics 输出，
        # 而 dataset.name 必须保持不变以保证数据加载和对抗样本路径正确。
        # 相对路径按项目根目录解析；不写的键仍使用上面的默认路径。
        for _k, _v in (self.cfg.get('paths') or {}).items():
            if _k != 'output_root' and _v:
                self.paths[_k] = os.path.normpath(
                    _v if os.path.isabs(_v) else os.path.join(self._project_root, _v)
                )

        # 确保目录存在
        for p in self.paths.values():
            if p.endswith('.json'):
                os.makedirs(os.path.dirname(p), exist_ok=True)
            else:
                os.makedirs(p, exist_ok=True)

        # 权重路径：项目根目录下的 weights/
        wp = self.cfg['model']['weights_path']
        if not os.path.isabs(wp):
            self.weights_path = os.path.join(self._project_root, wp)
        else:
            self.weights_path = wp
        self.weights_path = os.path.normpath(self.weights_path)

        # 设备
        self.device = self.cfg['training_free']['device']

    def get(self, key, default=None):
        """点号访问：cfg.get('dream.K')"""
        keys = key.split('.')
        val = self.cfg
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, key, value):
        """点号设置：cfg.set('dream.B_max', 45.1234)"""
        keys = key.split('.')
        d = self.cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def save(self, yaml_path=None):
        """保存回 YAML（调参后自动写回 B_max 等）"""
        path = yaml_path or self._yaml_path
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def adv_path(self, attack_name, split):
        """对抗样本绝对路径：项目根目录/data/adversarial/fgsm_cifar100_val/adv_data.pkl"""
        return os.path.join(
            self._project_root,
            'data',
            'adversarial',
            f'{attack_name}_{self.dataset_name}_{split}',
            'adv_data.pkl'
        )

    def metrics_path(self, filename):
        os.makedirs(self.paths['metrics'], exist_ok=True)
        return os.path.join(self.paths['metrics'], filename)

    def grid_search_path(self, filename):
        os.makedirs(self.paths['grid_search'], exist_ok=True)
        return os.path.join(self.paths['grid_search'], filename)


def load_config(yaml_path):
    return Config(yaml_path)