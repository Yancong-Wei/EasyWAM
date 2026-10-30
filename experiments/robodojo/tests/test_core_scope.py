"""CPU-only regression checks for the RoboDojo development boundary."""
import ast
from pathlib import Path
import re
import unittest
from urllib.parse import unquote

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
REMOVED = {'libero', 'libero_plus', 'libero_pro', 'robocasa', 'robotwin', 'vlabench'}


class CoreScopeTest(unittest.TestCase):
    def test_only_robodojo_benchmark_configs_and_sources(self):
        self.assertEqual({p.stem for p in (ROOT/'configs/data').glob('*.yaml')}, {'robodojo'})
        self.assertEqual({p.stem for p in (ROOT/'configs/benchmark').glob('*.yaml')}, {'sim_robodojo'})
        for path in (ROOT/'configs/task').glob('*.yaml'):
            self.assertTrue(path.name.startswith('robodojo_'), path)
        for name in REMOVED:
            self.assertFalse(list((ROOT/'experiments'/name).rglob('*.py')), name)
        self.assertFalse((ROOT/'configs/task/robodojo_xiconwam_baseline_wan22_lora.yaml').exists())

    def test_all_retained_task_and_model_configs_resolve(self):
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
            for task in sorted((ROOT/'configs/task').glob('*.yaml')):
                with self.subTest(task=task.stem):
                    cfg = compose(config_name='train', overrides=[f'task={task.stem}'])
                    OmegaConf.to_container(cfg, resolve=True)
                    self.assertEqual(cfg.data.train.processor.action_output_dim, 14)
            for model in sorted((ROOT/'configs/model').glob('*.yaml')):
                with self.subTest(model=model.stem):
                    cfg = compose(config_name='train', overrides=[
                        'task=robodojo_easywam_mot_wan22', f'model={model.stem}'])
                    OmegaConf.to_container(cfg, resolve=True)
                    module, factory = cfg.model._target_.rsplit('.', 1)
                    tree = ast.parse((ROOT/'src'/f'{module}.py').read_text())
                    self.assertIn(factory, {node.name for node in tree.body if isinstance(node, ast.FunctionDef)})

    def test_core_imports_do_not_reach_removed_benchmarks(self):
        paths = [p for folder in ('src', 'scripts', 'experiments') for p in (ROOT/folder).rglob('*.py')]
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else (
                    [name.name for name in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    self.assertFalse(any(name == f'experiments.{removed}' or
                        name.startswith(f'experiments.{removed}.') for removed in REMOVED), (path, name))
                    if name.startswith('experiments.'):
                        parts = name.split('.')
                        target = ROOT.joinpath(*parts)
                        self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(), (path, name))

    def test_document_links_have_existing_local_targets(self):
        paths = list((ROOT/'docs').rglob('*.md')) + [ROOT/'README.md', ROOT/'README_zh.md', ROOT/'CONTRIBUTING.md']
        failures = []
        for path in paths:
            for raw in re.findall(r'\]\(([^)]+)\)', path.read_text()):
                link = raw.split(' "', 1)[0].strip('<>')
                if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', link) or link.startswith('#'):
                    continue
                target = unquote(link.split('#', 1)[0])
                if target and not (path.parent/target).exists():
                    failures.append((str(path.relative_to(ROOT)), target))
        self.assertEqual(failures, [])

    def test_historical_camera_layout_contract_is_preserved(self):
        cfg = OmegaConf.load(ROOT/'configs/data/robodojo.yaml')
        self.assertEqual(cfg.train.concat_multi_camera, 'robotwin')
        self.assertEqual(list(cfg.train.video_size), [384, 320])
        self.assertEqual(cfg.train.processor.action_output_dim, 14)
        self.assertEqual(cfg.val.pretrained_norm_stats, cfg.train.pretrained_norm_stats)


if __name__ == '__main__':
    unittest.main()
