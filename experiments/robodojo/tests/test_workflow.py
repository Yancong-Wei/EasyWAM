import json
import tempfile
import unittest
import argparse
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from experiments.robodojo.result_utils import aggregate_rows, summarize_jobs, result_parent
from experiments.robodojo.policy_server import _load_config
from experiments.robodojo.run_identity import evaluation_identity, bind_output, evaluation_lock
from experiments.robodojo.upstream import validate_robodojo_root
from experiments.robodojo.run_robodojo_manager import _model_overrides, _select_jobs, _resolve_dataset_stats_path
from scripts.robodojo_pipeline import ROOT, TASK, check_data, main, download


def row(name, dimension, seed=0, score=50, complete=True, episodes=50):
    return dict(task_name=name, dimension=dimension, seed=seed, score=score,
                success_rate=score, complete=complete, episodes=episodes)


class WorkflowTest(unittest.TestCase):
    def test_planning_needs_no_policy_submodule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('scripts/eval_policy.sh', 'scripts/internal/task_inventory.py',
                         'scripts/internal/summarize_result.py', 'src/eval_client/main.py', 'env_cfg/arx_x5.yml'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test')
            self.assertEqual(validate_robodojo_root(root, require_policy=False), root.resolve())
            with self.assertRaises(FileNotFoundError):
                validate_robodojo_root(root)

    def test_full_config_no_lora(self):
        with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base='1.3'):
            cfg = compose(config_name='train', overrides=[f'task={TASK}'])
        self.assertEqual(cfg.model._target_, 'runtime.create_easywam_mot')
        self.assertNotIn('lora', cfg.model)
        self.assertEqual(cfg.model.action_dit_config.action_dim, 14)
        self.assertEqual(cfg.model.backbone.name, 'wan22')
        self.assertTrue(cfg.model.action_dit_config.use_gradient_checkpointing)
        self.assertEqual(cfg.batch_size * cfg.gradient_accumulation_steps, 16)

    def test_default_training_and_inference_recipe_match(self):
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            train=compose(config_name='train',overrides=[f'task={TASK}'])
            infer=compose(config_name='benchmark/sim_robodojo',overrides=[f'task={TASK}'])
        training_model=OmegaConf.to_container(train.model,resolve=True)
        inference_model=OmegaConf.to_container(infer.model,resolve=True)
        for config in (training_model, inference_model):
            config.pop('skip_dit_load_from_pretrain')
            config.pop('action_dit_pretrained_path')
            config['backbone'].pop('load_text_encoder')
        self.assertEqual(training_model,inference_model)
        self.assertEqual(OmegaConf.to_container(train.data.train.processor,resolve=True),
                         OmegaConf.to_container(infer.data.train.processor,resolve=True))
        self.assertEqual(train.data.train.pretrained_norm_stats,infer.data.train.pretrained_norm_stats)

    def test_partial_has_no_headline(self):
        result = aggregate_rows([row('a','precision'), row('b','memory',complete=False)])
        self.assertIsNone(result['overall']['score'])
        self.assertFalse(result['overall']['full_protocol_complete'])

    def test_dimension_weighting_not_pooled(self):
        rows = [row('a','precision',score=100), row('b','precision',score=100), row('c','memory',score=0)]
        result = aggregate_rows(rows)
        self.assertEqual(result['overall']['score'], 50)
        self.assertFalse(result['overall']['full_protocol_complete'])

    def test_generalization_needs_both_variants(self):
        result = aggregate_rows([row('a','generalization',episodes=25)])
        self.assertFalse(result['overall']['complete'])
        result = aggregate_rows([row('a','generalization',episodes=25,score=100),row('a_random','generalization',episodes=25,score=0)])
        self.assertEqual(result['overall']['score'], 50)

    def test_seed_std_and_full_coverage(self):
        rows=[]
        for seed in range(3):
            for index in range(42):
                dim = 'generalization' if index < 12 else ['precision','memory','long','open'][index % 4]
                rows.append(row(f't{index}',dim,seed,seed*50,episodes=25 if index < 12 else 50))
                if index < 12:
                    rows.append(row(f't{index}_random',dim,seed,seed*50,episodes=25))
        result=aggregate_rows(rows)
        self.assertTrue(result['overall']['full_protocol_complete'])
        self.assertFalse(result['overall']['officially_verified'])
        self.assertAlmostEqual(result['overall']['score_std'], (5000/3)**0.5)
        self.assertEqual(result['overall']['score'],50)

    def test_duplicate_cells_rejected(self):
        with self.assertRaises(ValueError): aggregate_rows([row('a','memory'),row('a','memory')])

    def test_checkpoint_and_settings_change_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); ckpt=root/'weights.pt'; stats=root/'stats.json'
            ckpt.write_bytes(b'one'); stats.write_text('{}')
            a=evaluation_identity(ckpt,stats,{'steps':10})
            ckpt.write_bytes(b'two'); b=evaluation_identity(ckpt,stats,{'steps':10})
            c=evaluation_identity(ckpt,stats,{'steps':5})
            self.assertNotEqual(a['tag'],b['tag']); self.assertNotEqual(b['tag'],c['tag'])
            bind_output(root,a); bind_output(root,a)
            with self.assertRaises(ValueError): bind_output(root,b)

    def test_output_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with evaluation_lock(Path(tmp)):
                with self.assertRaises(RuntimeError):
                    with evaluation_lock(Path(tmp)): pass

    def test_eval_overrides_forward_resolved_config(self):
        with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
            cfg=compose(config_name='benchmark/sim_robodojo',overrides=[f'task={TASK}'])
        args=_model_overrides(cfg,Path('/c.pt'),Path('/stats.json'),Path('/resolved.yaml'))
        self.assertIn('--resolved_sim_cfg_path', args)
        self.assertIn("'/resolved.yaml'",args)

    def test_manager_to_server_override_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix='policy path ') as tmp:
            root = Path(tmp)
            with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
                cfg=compose(config_name='benchmark/sim_robodojo',overrides=[f'task={TASK}',
                    f'EVALUATION.output_dir={json.dumps(str(root))}',
                    'model.backbone.dit_config.video_attention_mask_mode=bidirectional',
                    'EVALUATION.replan_steps=8'])
            resolved = root/'resolved config.yaml'
            OmegaConf.save(cfg, resolved, resolve=True)
            args = _model_overrides(cfg, root/'weights.pt', root/'stats.json', resolved)
            parsed = _load_config(ROOT/'experiments/robodojo/easywam_policy/deploy_policy.yml', args)
            restored = OmegaConf.load(parsed['resolved_sim_cfg_path'])
            self.assertEqual(parsed['replan_steps'], 8)
            self.assertEqual(restored.model.backbone.dit_config.video_attention_mask_mode, 'bidirectional')
            self.assertTrue(restored.model.skip_dit_load_from_pretrain)
            self.assertTrue(restored.model.backbone.load_text_encoder)
            self.assertIsNone(restored.model.action_dit_pretrained_path)

    def test_raw_results_to_scores_and_incomplete_latest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [dict(task_name=name,dimension='generalization',seed=0,episodes=2)
                    for name in ['a','a_random']]
            for job in jobs:
                folder = result_parent(root, job['task_name'], 'arx_x5', 0, 'identity')/'2026-01-01'
                folder.mkdir(parents=True)
                value = 1.0 if job['task_name']=='a' else 0.0
                (folder/'_result.json').write_text(json.dumps({'eval_time':2, 'details':{
                    str(i): {'success':bool(value),'score':value} for i in range(2)}}))
            result=aggregate_rows(summarize_jobs(jobs,root,'arx_x5','identity'))
            self.assertEqual(result['overall']['score'],50)
            latest=result_parent(root,'a','arx_x5',0,'identity')/'2026-01-02'
            latest.mkdir()
            (latest/'_result.json').write_text('{broken')
            result=aggregate_rows(summarize_jobs(jobs,root,'arx_x5','identity'))
            self.assertFalse(result['overall']['complete'])
            self.assertIsNone(result['overall']['score'])

    def test_training_dry_run_never_executes(self):
        with patch('scripts.robodojo_pipeline.subprocess.run') as run:
            main(['probe','--run-id','test','--dry-run'])
            run.assert_not_called()

    def test_lora_override_rejected(self):
        with self.assertRaises(ValueError):
            main(['train','--run-id','test','--dry-run','model.lora.r=32'])

    def test_wrong_data_action_space_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'meta').mkdir()
            (root/'meta/info.json').write_text(json.dumps({'features':{'action':{'shape':[16]}}}))
            (root/'dataset_stats.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'joint14'): check_data(root)

    def test_incomplete_download_cannot_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'easywam_download.json').write_text(json.dumps({'complete': False}))
            with self.assertRaisesRegex(RuntimeError, 'Download is incomplete'):
                check_data(root)

    def test_generalization_smoke_task_selection(self):
        records=[dict(name='a',dimension='generalization'),dict(name='a_random',dimension='generalization')]
        jobs=_select_jobs(records,'a',[0],1)
        self.assertEqual(len(jobs),2)
        self.assertEqual([job['episodes'] for job in jobs],[1,1])

    def test_explicit_missing_stats_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'dataset_stats.json').write_text('{}')
            cfg = OmegaConf.create({'EVALUATION': {'dataset_stats_path': str(root / 'missing.json')}})
            with self.assertRaises(FileNotFoundError):
                _resolve_dataset_stats_path(cfg, root / 'weights.pt')

    def test_disk_guard_stops_before_snapshot_download(self):
        from unittest.mock import Mock
        hub = SimpleNamespace(HfApi=Mock(), snapshot_download=Mock())
        hub.HfApi.return_value.repo_info.return_value = SimpleNamespace(
            sha='pinned', siblings=[SimpleNamespace(rfilename='huge.bin', size=111 * 1024**3)])
        args = argparse.Namespace(asset='dataset', dry_run=False, endpoint=None, revision='main',
                                  reserve_gib=30, size_only=False, workers=1)
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.robodojo_pipeline.DATA', Path(tmp)), \
             patch.dict(sys.modules, {'huggingface_hub': hub}), \
             patch('scripts.robodojo_pipeline.shutil.disk_usage', return_value=SimpleNamespace(free=50 * 1024**3)):
            with self.assertRaisesRegex(RuntimeError, 'Insufficient disk'):
                download(args)
        hub.snapshot_download.assert_not_called()


if __name__=='__main__': unittest.main()
